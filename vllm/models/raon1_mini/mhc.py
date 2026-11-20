# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manifold-constrained hyper-connections — RAON's depth mechanism.

Port of the ``raon-mhc`` arm of Megatron's
``megatron/core/transformer/mhc_residual.py`` (``RaonMHCRes``,
``MHCStreamResidual``) to vLLM's token-flat inference layout.

Per token, with X the n residual streams stacked as (n, C)::

    logits = (vec(X) phi^T) * rms_inv(vec(X))     # scale AFTER the GEMM
    H_pre  = sigmoid(alpha_pre  logits[0:n]   + b_pre)          (n,)
    H_post = 2 sigmoid(alpha_post logits[n:2n] + b_post)        (n,)
    H_res  = (1 - g) I + g Phi_B(Q Z Q^T),  g = sigmoid(eta)    (n, n)
    X_next = H_res X + H_post (x) F(H_pre X)

Nothing persists across decode steps and nothing goes in the kv cache: every
site reads only that token's own (n, C) slice. The stream tensor is an
ordinary activation threaded through the layer loop.

The branch F carries NO residual of its own. The entire residual lives inside
H_res; adding vLLM's usual ``hidden = residual + attn_out`` anywhere in this
model produces a different network.
"""

from __future__ import annotations
import functools
import math
import os
from typing import Callable
import torch
from torch import Tensor, nn
from . import raon_phase
from .batch_invariant import matmul_persistent

_EPS = 1e-05
_FUSED = os.environ.get("RAON_MHC_FUSED", "1") == "1"
_CHECK = os.environ.get("RAON_MHC_CHECK", "0") == "1"
_STAGES1 = os.environ.get("RAON_MHC_STAGES1", "0") == "1"
_HEAD_RMS = os.environ.get("RAON_MHC_HEAD_RMS", "0") == "1"
_HEAD_CHART = os.environ.get("RAON_MHC_HEAD_CHART", "0") == "1"
_SPLIT_MAX_T = int(os.environ.get("RAON_MHC_SPLIT_MAX_T", "0"))
_SPLIT_MIX_BC = int(os.environ.get("RAON_MHC_SPLIT_MIX_BC", "512"))
_SPLIT_HEAD_BC = int(os.environ.get("RAON_MHC_SPLIT_HEAD_BC", "512"))
_SGL_STATS = {"head_v2": 0, "rms_in": 0, "chart_in": 0, "split": 0}
_SGL_ANNOUNCED = False
_BATCH_INVARIANT = os.environ.get("RAON_BATCH_INVARIANT", "1") == "1"
_MULTI_TOKEN_MIX = os.environ.get("RAON_MHC_MULTI_TOKEN_MIX", "0") == "1"
_MULTI_TOKEN_MIX_SEEN = set()


def _launch_site_mix(
    grid, s, y, head, e, h0, neg_h0, g_eye, g_gate, out, *, C, N, MIX_ONE, **kwargs
):
    if _MULTI_TOKEN_MIX and s.shape[0] >= 4096 and (C == 2560) and (N == 4):
        from .mhc_site_mix import mix_site

        tokens = s.shape[0]
        if tokens not in _MULTI_TOKEN_MIX_SEEN and len(_MULTI_TOKEN_MIX_SEEN) < 64:
            _MULTI_TOKEN_MIX_SEEN.add(tokens)
            print(
                f"[mhc-multi-token-mix] engaged T={tokens},tile=(8,128,4)", flush=True
            )
        mix_site(
            s,
            y,
            head,
            e,
            h0,
            neg_h0,
            g_eye,
            g_gate,
            output=out,
            tile=(8, 128, 4),
            mix_one=MIX_ONE,
        )
        return
    _mhc_site_mix[grid](
        s,
        y,
        head,
        e,
        h0,
        neg_h0,
        g_eye,
        g_gate,
        out,
        C=C,
        N=N,
        MIX_ONE=MIX_ONE,
        **kwargs,
        enable_fp_fusion=False,
    )


_FIXED_AGG_TREE = False
_FIXED_MIX_ONE = False
_TRAINER_FUSED_ENV = os.environ.get("RAON_MHC_TRAINER_FUSED", "0") == "1"
_TRAINER_FUSED = _TRAINER_FUSED_ENV or raon_phase.PREFILL_ALIGNED
_PHI_DIRECT_ENV = "RAON_MHC_PHI_GEMM_DIRECT"
_PHI_DIRECT_N_PAD = 64
_PHI_DIRECT_M = 32


def _phi_direct_mode() -> str:
    raw = os.environ.get(_PHI_DIRECT_ENV, "0").strip().lower()
    if raw in ("0", "off", ""):
        return "off"
    if raw == "1":
        return "m32"
    if raw in ("all", "all-pad"):
        return raw
    raise RuntimeError(
        f"{_PHI_DIRECT_ENV}={raw!r} is not one of ('0', 'off', '', '1', 'all', 'all-pad'). Refusing to guess."
    )


_PHI_DIRECT_MODE = _phi_direct_mode()
_PHI_DIRECT = _PHI_DIRECT_MODE != "off"
_PHI_ALL_TIERS = {
    "all": (
        (8, "direct", (256, 1, 1), 17),
        (16, "direct", (256, 1, 2), 17),
        (64, "direct", (256, 2, 2), 18),
        (128, "direct", (128, 2, 4), 18),
        (256, "direct", (64, 2, 4), 18),
        (512, "direct", (64, 2, 8), 18),
        (1 << 62, "direct", (128, 4, 4), 20),
    )
}
_PHI_ALL_TIERS["all-pad"] = _PHI_ALL_TIERS["all"][:5] + (
    (1024, "padmm", None, 24),
    (1 << 62, "padmm", None, 32),
)
if _PHI_DIRECT:
    from .phi_direct_gemm_vendored import (
        default_tactic as _phi_default_tactic,
        run_direct_dense as _phi_run_direct_dense,
    )
if _PHI_DIRECT_MODE in ("all", "all-pad"):
    from .phi_direct_gemm_dynm import (
        DynMTactic as _PhiDynMTactic,
        run_direct_dense_dynm as _phi_run_dynm,
    )
_PHI_DIRECT_STATS = {"direct": 0, "ineligible": 0, "error": 0}
_PHI_DIRECT_ERR_PRINTED = False
_PHI_DIRECT_CHECKED = False


def _phi_direct_mm(site, v: "Tensor") -> "Tensor":
    """Direct-kernel phi logits GEMM with a loud, counted fallback.

    Returns the (T, 64) padded output when the direct kernel served the call
    (or its error fallback on the padded weight would -- see below), and the
    (T, 17) unpadded torch.mm output when the shape is ineligible. Callers
    read the returned tensor's column count; both consumption sites handle
    either width.

    Eligibility is exactly ``M == 32`` (the captured decode shape; the
    compile cache for that single (dtype, m, n, k, tactic, pdl) key is charged
    in finalize(), so a captured call can never JIT). Every other M --
    prefill, drafts, tails -- takes the unpadded torch.mm, byte-identical to
    the lever-off engine.
    """
    global _PHI_DIRECT_ERR_PRINTED
    if _PHI_DIRECT_MODE != "m32":
        return _phi_all_mm(site, v)
    if v.shape[0] != _PHI_DIRECT_M:
        _PHI_DIRECT_STATS["ineligible"] += 1
        return torch.mm(v, site.phi.t())
    try:
        out = torch.empty(
            _PHI_DIRECT_M, _PHI_DIRECT_N_PAD, device=v.device, dtype=v.dtype
        )
        _phi_run_direct_dense(
            v, site.phi_padded.t(), out, pdl=False, tactic=site._phi_direct_tactic
        )
        _PHI_DIRECT_STATS["direct"] += 1
        return out
    except Exception as exc:
        _PHI_DIRECT_STATS["error"] += 1
        if not _PHI_DIRECT_ERR_PRINTED:
            _PHI_DIRECT_ERR_PRINTED = True
            print(
                f"[va] {_PHI_DIRECT_ENV}=1 direct GEMM RAISED and fell back to torch.mm (unpadded): {exc!r}. Counted in _PHI_DIRECT_STATS; the error count should be 0.",
                flush=True,
            )
        return torch.mm(v, site.phi.t())


def _phi_direct_self_check(site, v: "Tensor") -> None:
    """First-live-forward self-check (same pattern as moe_router_gemm.py
    `_pad8_self_check`), on the real activations and weights, at the eligible
    shape only.

    Two claims:
    1. FAIL-CLOSED -- the padded weight rows are zero, so the direct output's
       padded columns [17:64] must be exactly zero. A nonzero means
       phi_padded is stale/corrupt (bad refit?) and the lever is corrupting
       the site: raise. A raise from the direct kernel itself also propagates
       here (deterministic breakage must fail loudly, not hide in the
       fallback counter).
    2. RECORDED -- direct[:, :17] vs torch.mm(v, phi.t()) bitwise (bf16). A
       difference is reported as ULP-class (summation order), not raised.

    Skipped (without latching) while a CUDA stream capture is active: the
    check syncs the host (`int(...)`), which is illegal under capture, and
    the eager warmup pass before FDO capture is where it is meant to fire.
    """
    global _PHI_DIRECT_CHECKED
    if _PHI_DIRECT_MODE != "m32":
        return _phi_all_self_check(site, v)
    if _PHI_DIRECT_CHECKED or v.shape[0] != _PHI_DIRECT_M:
        return
    if v.is_cuda and torch.cuda.is_current_stream_capturing():
        return
    _PHI_DIRECT_CHECKED = True
    R = site.phi.shape[0]
    with torch.no_grad():
        y0 = torch.mm(v, site.phi.t())
        yd = torch.empty(
            _PHI_DIRECT_M, _PHI_DIRECT_N_PAD, device=v.device, dtype=v.dtype
        )
        _phi_run_direct_dense(
            v, site.phi_padded.t(), yd, pdl=False, tactic=site._phi_direct_tactic
        )
        tail_nonzero = int((yd[:, R:] != 0).sum())
        if tail_nonzero:
            raise RuntimeError(
                f"{_PHI_DIRECT_ENV}=1 self-check FAILED: {tail_nonzero} nonzero elements in the padded output columns [{R}:{_PHI_DIRECT_N_PAD}]. The zero weight rows leaked, so phi_padded does not hold the live phi plus zeros (stale after a refit?). Refusing to run."
            )
        real = yd[:, :R].contiguous()
        neq = int((real.view(torch.int16) != y0.view(torch.int16)).sum())
        if neq:
            d = (real.float() - y0.float()).abs()
            md = float(d.max())
            print(
                f"[va] {_PHI_DIRECT_ENV}=1 self-check: direct[:, :{R}] is NOT bit-identical to torch.mm -- {neq}/{y0.numel()} bf16 logits differ (max|d| {md:.3e}, T={v.shape[0]}). ULP-class, not bitwise-safe; do not enable where bit parity is required. stats={_PHI_DIRECT_STATS}",
                flush=True,
            )
        else:
            print(
                f"[va] {_PHI_DIRECT_ENV}=1 self-check PASS: direct[:, :{R}] == torch.mm(v, phi.t()) bitwise ({y0.numel()} bf16 logits, T={v.shape[0]}) and the padded tail is exactly zero on the live activations/weight. tactic={site._phi_direct_tactic} stats={_PHI_DIRECT_STATS}",
                flush=True,
            )


_PHI_ALL_SEEN: set = set()
_PHI_ALL_CHECKED: set = set()
_PHI_ALL_CALLS = 0


def _phi_all_tier(m: int):
    for tier in _PHI_ALL_TIERS[_PHI_DIRECT_MODE]:
        if m <= tier[0]:
            return tier
    raise AssertionError("unreachable: the last tier is unbounded")


def _phi_tier_name(tier) -> str:
    hi, kind, tac, npad = tier
    t = "" if tac is None else "-" + "-".join(map(str, tac))
    return f"{kind}{t}-n{npad}@M<={(hi if hi < 1 << 62 else 'inf')}"


def _phi_all_run(site, v: "Tensor", tier) -> "Tensor":
    """The tier's GEMM, no ledger. (T, npad) output, real columns [:, :R]."""
    _, kind, tac, npad = tier
    if kind == "mm":
        return torch.mm(v, site.phi.t())
    w = (
        site.phi.detach()
        if npad == site.phi.shape[0]
        else getattr(site, f"phi_pad{npad}")
    )
    if kind == "padmm":
        return torch.mm(v, w.t())
    out = torch.empty(v.shape[0], npad, device=v.device, dtype=v.dtype)
    _phi_run_dynm(v, w.t(), out, _PhiDynMTactic(*tac))
    return out


def _phi_all_mm(site, v: "Tensor") -> "Tensor":
    """Tiered phi logits GEMM with a loud, counted fallback (same contract as
    _phi_direct_mm: the caller reads the returned column count)."""
    global _PHI_DIRECT_ERR_PRINTED, _PHI_ALL_CALLS
    tier = _phi_all_tier(v.shape[0])
    name = _phi_tier_name(tier)
    try:
        out = _phi_all_run(site, v, tier)
    except Exception as exc:
        _PHI_DIRECT_STATS["error"] += 1
        if not _PHI_DIRECT_ERR_PRINTED:
            _PHI_DIRECT_ERR_PRINTED = True
            print(
                f"[phi-all] {_PHI_DIRECT_ENV}={_PHI_DIRECT_MODE} tier {name} RAISED at M={v.shape[0]} and fell back to torch.mm (unpadded): {exc!r}. Counted in _PHI_DIRECT_STATS; the error count should be 0.",
                flush=True,
            )
        return torch.mm(v, site.phi.t())
    _PHI_DIRECT_STATS[name] = _PHI_DIRECT_STATS.get(name, 0) + 1
    _PHI_ALL_CALLS += 1
    first = name not in _PHI_ALL_SEEN
    if first or _PHI_ALL_CALLS & _PHI_ALL_CALLS - 1 == 0:
        _PHI_ALL_SEEN.add(name)
        cap = v.is_cuda and torch.cuda.is_current_stream_capturing()
        print(
            f"[phi-all] {_PHI_DIRECT_ENV}={_PHI_DIRECT_MODE} {('first use of ' + name if first else 'ledger')} M={v.shape[0]} capturing={cap} calls={_PHI_ALL_CALLS} stats={_PHI_DIRECT_STATS}",
            flush=True,
        )
    return out


def _phi_all_self_check(site, v: "Tensor") -> None:
    """Once per non-"mm" tier, on its first eager call: padded output columns
    must be exactly zero (else raise -- stale/corrupt padded weight), and the
    real columns are compared bitwise with the unpadded torch.mm (recorded,
    not judged: ~3e-4 of the logits differ by fp32 summation order even where
    the direct kernel is the more accurate one -- see the tier table)."""
    tier = _phi_all_tier(v.shape[0])
    name = _phi_tier_name(tier)
    if tier[1] == "mm" or name in _PHI_ALL_CHECKED:
        return
    if v.is_cuda and torch.cuda.is_current_stream_capturing():
        return
    _PHI_ALL_CHECKED.add(name)
    R = site.phi.shape[0]
    with torch.no_grad():
        y0 = torch.mm(v, site.phi.t())
        yd = _phi_all_run(site, v, tier)
        if yd.shape[1] > R:
            tail_nonzero = int((yd[:, R:] != 0).sum())
            if tail_nonzero:
                raise RuntimeError(
                    f"{_PHI_DIRECT_ENV}={_PHI_DIRECT_MODE} self-check FAILED ({name}): {tail_nonzero} nonzero elements in the padded output columns [{R}:{yd.shape[1]}]. phi_pad{yd.shape[1]} does not hold the live phi plus zeros (stale after a refit?). Refusing to run."
                )
        real = yd[:, :R].contiguous()
        neq = int((real.view(torch.int16) != y0.view(torch.int16)).sum())
        md = float((real.float() - y0.float()).abs().max()) if neq else 0.0
        print(
            f"[phi-all] {_PHI_DIRECT_ENV}={_PHI_DIRECT_MODE} self-check {('PASS' if not neq else 'DIFF')} {name} M={v.shape[0]}: {neq}/{y0.numel()} bf16 logits differ from torch.mm(v, phi.t()) (max|d| {md:.3e}); padded tail zero.",
            flush=True,
        )


def _phi_all_finalize(site, R: int) -> None:
    """Build the padded weights the tiers need (rebuilt from the live phi on
    every finalize, i.e. on refit) and precompile every direct tactic before
    any CUDA-graph capture (one compile per tactic serves all M)."""
    tiers = _PHI_ALL_TIERS[_PHI_DIRECT_MODE]
    K = site.phi.shape[1]
    for npad in sorted({t[3] for t in tiers if t[3] != R}):
        pp = torch.zeros(npad, K, dtype=site.phi.dtype, device=site.phi.device)
        pp[:R] = site.phi.detach()
        site._refit_set_buf(f"phi_pad{npad}", pp)
    if not site.phi.is_cuda:
        return
    with torch.no_grad():
        for tier in tiers:
            if tier[1] == "direct":
                m = min(tier[0], 128)
                zin = torch.zeros(m, K, dtype=site.phi.dtype, device=site.phi.device)
                _phi_all_run(site, zin, tier)
    if not getattr(_phi_all_finalize, "_printed", False):
        _phi_all_finalize._printed = True
        print(
            f"[phi-all] {_PHI_DIRECT_ENV}={_PHI_DIRECT_MODE} armed: tiers {[_phi_tier_name(t) for t in tiers]}; direct tactics precompiled (finalize).",
            flush=True,
        )


_AGGNORM_FUSED_ENV = os.environ.get("RAON_MHC_AGGNORM_FUSED", "0") == "1"
_AGGNORM_FUSED = _AGGNORM_FUSED_ENV or raon_phase.PREFILL_ALIGNED
try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice as tld
except ImportError:
    _FUSED = False
    _TRAINER_FUSED = False
    _TRAINER_FUSED_ENV = False
    _AGGNORM_FUSED = False
    _AGGNORM_FUSED_ENV = False
if _AGGNORM_FUSED and (not _TRAINER_FUSED):
    raise RuntimeError(
        "RAON_MHC_AGGNORM_FUSED=1 requires RAON_MHC_TRAINER_FUSED=1: the fused aggregate-norm replaces the trainer-fused site's plain aggregate; there is no aggregate-norm arm on the other engine paths (a silent no-op arm would lie in the parity manifest)"
    )


def helmert_basis(n: int, dtype: torch.dtype = torch.float32) -> Tensor:
    """Orthonormal Q in R^{n x (n-1)} with Q^T Q = I and Q^T 1 = 0."""
    q = torch.zeros(n, n - 1, dtype=dtype)
    for k in range(1, n):
        q[:k, k - 1] = 1.0
        q[k, k - 1] = -float(k)
        q[:, k - 1] /= math.sqrt(k * (k + 1.0))
    return q


def raon_mhc_gamma_init(n: int) -> float:
    """Radius that reproduces DeepSeek mHC's {0 diag, -8 off} Sinkhorn init."""
    e = math.exp(-8.0)
    return 2.0 * n * e / (1.0 + (n - 1) * e)


def h_res_chart(z: Tensor, eta: Tensor, q: Tensor, h0: Tensor, eye: Tensor) -> Tensor:
    """Chart coordinates Z (M, n-1, n-1) -> doubly stochastic H_res (M, n, n).

    Runs entirely in ``z.dtype``: fp32 in the model, fp64 in test_mhc.py where
    double stochasticity is checkable at 1e-12. An eps'd normalization here
    contracts the streams ~1e-5 per site and the top-k router turns that into
    O(1) output changes at routing boundaries, so the clamp floor is exactly
    0.0 and there is no eps anywhere in this function.
    """
    e = torch.einsum("im,bmk,jk->bij", q, z, q)
    rho = (-e / h0).amax(dim=(-2, -1)).clamp_min(0.0)
    phi = h0 + e / (1.0 + rho)[:, None, None]
    g = torch.sigmoid(eta)
    return (1.0 - g) * eye + g * phi


if _FUSED:

    @triton.jit
    def _mhc_square(s_ptr, sq_ptr, total, B: tl.constexpr):
        o = tl.program_id(0) * B + tl.arange(0, B)
        m = o < total
        v = tl.load(s_ptr + o, mask=m, other=0.0).to(tl.float32)
        tl.store(sq_ptr + o, v * v, mask=m)

    @triton.jit
    def _mhc_site_head(
        s_ptr,
        mm_ptr,
        mean_ptr,
        arow_ptr,
        brow_ptr,
        head_ptr,
        z_ptr,
        agg_ptr,
        NC: tl.constexpr,
        C: tl.constexpr,
        N: tl.constexpr,
        R: tl.constexpr,
        MMS: tl.constexpr,
        BC: tl.constexpr,
        EPS: tl.constexpr,
        AGG_TREE: tl.constexpr,
    ):
        t = tl.program_id(0)
        v_base = s_ptr + t * NC
        r = tl.arange(0, 32)
        rmask = r < R
        acc = tl.load(mm_ptr + t * MMS + r, mask=rmask, other=0.0).to(tl.float32)
        inv = tld.rsqrt(tl.load(mean_ptr + t) + EPS)
        a = tl.load(arow_ptr + r, mask=rmask, other=0.0)
        b = tl.load(brow_ptr + r, mask=rmask, other=0.0)
        pre = tl.add(tl.mul(a, tl.mul(acc, inv)), b)
        sig = tl.div_rn(tl.full(pre.shape, 1.0, tl.float32), 1.0 + tld.exp(-pre))
        hvec = tl.where(r < N, sig, 2.0 * sig)
        hb = head_ptr + t * (2 * N)
        tl.store(hb + r, hvec, mask=r < 2 * N)
        tl.store(z_ptr + t * (R - 2 * N) + (r - 2 * N), pre, mask=rmask & (r >= 2 * N))
        tl.debug_barrier()
        h0 = tl.load(hb + 0)
        h1 = tl.load(hb + 1)
        h2 = tl.load(hb + 2)
        h3 = tl.load(hb + 3)
        for c0 in range(0, C, BC):
            cols = c0 + tl.arange(0, BC)
            s0 = tl.load(v_base + 0 * C + cols).to(tl.float32)
            s1 = tl.load(v_base + 1 * C + cols).to(tl.float32)
            s2 = tl.load(v_base + 2 * C + cols).to(tl.float32)
            s3 = tl.load(v_base + 3 * C + cols).to(tl.float32)
            if AGG_TREE:
                p01 = tld.fma(h1, s1, tl.mul(h0, s0))
                p23 = tld.fma(h3, s3, tl.mul(h2, s2))
                aggv = tl.add(p01, p23)
            else:
                aggv = tld.fma(h3, s3, tld.fma(h2, s2, tld.fma(h1, s1, tl.mul(h0, s0))))
            tl.store(agg_ptr + t * C + cols, aggv.to(agg_ptr.dtype.element_ty))

    @triton.jit
    def _mhc_site_mix(
        s_ptr,
        y_ptr,
        head_ptr,
        e_ptr,
        h0_ptr,
        negh0_ptr,
        geye_ptr,
        ggate_ptr,
        out_ptr,
        C: tl.constexpr,
        N: tl.constexpr,
        BC: tl.constexpr,
        MIX_ONE: tl.constexpr,
    ):
        t = tl.program_id(0)
        cols = tl.program_id(1) * BC + tl.arange(0, BC)
        i4 = tl.arange(0, 4)
        j4 = tl.arange(0, 4)
        e = tl.load(e_ptr + t * N * N + i4[:, None] * N + j4[None, :])
        negh0 = tl.load(negh0_ptr + i4[:, None] * N + j4[None, :])
        rho = tl.maximum(tl.max(tl.max(tl.div_rn(e, negh0), axis=1), axis=0), 0.0)
        h0 = tl.load(h0_ptr + i4[:, None] * N + j4[None, :])
        phim = h0 + tl.div_rn(e, tl.full(e.shape, 1.0, tl.float32) + rho)
        geye = tl.load(geye_ptr + i4[:, None] * N + j4[None, :])
        g = tl.load(ggate_ptr)
        hres = tl.add(geye, tl.mul(g, phim))
        hb = head_ptr + t * (2 * N)
        hpost = tl.load(hb + N + i4)
        y = tl.load(y_ptr + t * C + cols).to(tl.float32)
        h0 = tl.sum(tl.where(j4 == 0, hres, 0.0), axis=1)
        h1 = tl.sum(tl.where(j4 == 1, hres, 0.0), axis=1)
        h2 = tl.sum(tl.where(j4 == 2, hres, 0.0), axis=1)
        h3 = tl.sum(tl.where(j4 == 3, hres, 0.0), axis=1)
        s0 = tl.load(s_ptr + t * N * C + 0 * C + cols).to(tl.float32)
        s1 = tl.load(s_ptr + t * N * C + 1 * C + cols).to(tl.float32)
        s2 = tl.load(s_ptr + t * N * C + 2 * C + cols).to(tl.float32)
        s3 = tl.load(s_ptr + t * N * C + 3 * C + cols).to(tl.float32)
        if MIX_ONE:
            p0 = tl.mul(h0[:, None], s0[None, :])
            p1 = tl.mul(h1[:, None], s1[None, :])
            p2 = tl.mul(h2[:, None], s2[None, :])
            p3 = tl.mul(h3[:, None], s3[None, :])
            acc = tl.add(tl.add(tl.add(p0, p1), p2), p3)
        else:
            acc = tld.fma(
                h3[:, None],
                s3[None, :],
                tld.fma(
                    h2[:, None],
                    s2[None, :],
                    tld.fma(h1[:, None], s1[None, :], tl.mul(h0[:, None], s0[None, :])),
                ),
            )
        o = tl.add(acc, tl.mul(hpost[:, None], y[None, :]))
        tl.store(
            out_ptr + t * N * C + i4[:, None] * C + cols[None, :],
            o.to(out_ptr.dtype.element_ty),
        )

    @triton.jit
    def _mhc_halve_x(v, H: tl.constexpr, w: tl.constexpr):
        return tl.sum(tl.reshape(v, [H, 2, w // 2]), axis=1)

    @triton.jit
    def _mhc_halve_y(v, h: tl.constexpr):
        return tl.sum(tl.reshape(v, [2, h // 2]), axis=0)

    @triton.jit
    def _mhc_rms_mean(
        v_base, NC: tl.constexpr, H: tl.constexpr, W: tl.constexpr, FACTOR: tl.constexpr
    ):
        L = tl.arange(0, W)[None, :] + tl.arange(0, H)[:, None] * W
        a0 = tl.zeros([H, W], tl.float32)
        a1 = tl.zeros([H, W], tl.float32)
        a2 = tl.zeros([H, W], tl.float32)
        a3 = tl.zeros([H, W], tl.float32)
        for k in tl.static_range(NC // 2048):
            off = (L + k * 512) * 4
            v0 = tl.load(v_base + off + 0).to(tl.float32)
            v1 = tl.load(v_base + off + 1).to(tl.float32)
            v2 = tl.load(v_base + off + 2).to(tl.float32)
            v3 = tl.load(v_base + off + 3).to(tl.float32)
            a0 = tl.add(a0, tl.mul(v0, v0))
            a1 = tl.add(a1, tl.mul(v1, v1))
            a2 = tl.add(a2, tl.mul(v2, v2))
            a3 = tl.add(a3, tl.mul(v3, v3))
        v = tl.add(tl.add(tl.add(a0, a1), a2), a3)
        if W >= 512:
            v = _mhc_halve_x(v, H, 512)
        if W >= 256:
            v = _mhc_halve_x(v, H, 256)
        if W >= 128:
            v = _mhc_halve_x(v, H, 128)
        if W >= 64:
            v = _mhc_halve_x(v, H, 64)
        v = _mhc_halve_x(v, H, 32)
        v = _mhc_halve_x(v, H, 16)
        v = _mhc_halve_x(v, H, 8)
        v = _mhc_halve_x(v, H, 4)
        v = _mhc_halve_x(v, H, 2)
        v = tl.reshape(v, [H])
        if H >= 16:
            v = _mhc_halve_y(v, 16)
        if H >= 8:
            v = _mhc_halve_y(v, 8)
        if H >= 4:
            v = _mhc_halve_y(v, 4)
        if H >= 2:
            v = _mhc_halve_y(v, 2)
        return tl.mul(tl.sum(v, axis=0), FACTOR)

    @triton.jit
    def _mhc_dot3(a0, a1, a2, b0, b1, b2, FMA: tl.constexpr):
        if FMA:
            r = tld.fma(a2, b2, tld.fma(a1, b1, tl.mul(a0, b0)))
        else:
            r = tl.add(tl.add(tl.mul(a0, b0), tl.mul(a1, b1)), tl.mul(a2, b2))
        return r

    @triton.jit
    def _mhc_site_head_v2(
        s_ptr,
        mm_ptr,
        mean_ptr,
        arow_ptr,
        brow_ptr,
        head_ptr,
        z_ptr,
        e_ptr,
        q_ptr,
        agg_ptr,
        NC: tl.constexpr,
        C: tl.constexpr,
        N: tl.constexpr,
        R: tl.constexpr,
        MMS: tl.constexpr,
        BC: tl.constexpr,
        EPS: tl.constexpr,
        AGG_TREE: tl.constexpr,
        RMS_IN: tl.constexpr,
        RH: tl.constexpr,
        RW: tl.constexpr,
        FACTOR: tl.constexpr,
        CHART_IN: tl.constexpr,
        CHART_FMA2: tl.constexpr,
        SPLIT: tl.constexpr,
    ):
        t = tl.program_id(0)
        v_base = s_ptr + t * NC
        r = tl.arange(0, 32)
        rmask = r < R
        acc = tl.load(mm_ptr + t * MMS + r, mask=rmask, other=0.0).to(tl.float32)
        if RMS_IN:
            mean = _mhc_rms_mean(v_base, NC, RH, RW, FACTOR)
        else:
            mean = tl.load(mean_ptr + t)
        inv = tld.rsqrt(mean + EPS)
        a = tl.load(arow_ptr + r, mask=rmask, other=0.0)
        b = tl.load(brow_ptr + r, mask=rmask, other=0.0)
        pre = tl.add(tl.mul(a, tl.mul(acc, inv)), b)
        sig = tl.div_rn(tl.full(pre.shape, 1.0, tl.float32), 1.0 + tld.exp(-pre))
        hvec = tl.where(r < N, sig, 2.0 * sig)
        hb = head_ptr + t * (2 * N)
        tl.store(hb + r, hvec, mask=r < 2 * N)
        zb = z_ptr + t * (R - 2 * N)
        tl.store(zb + (r - 2 * N), pre, mask=rmask & (r >= 2 * N))
        tl.debug_barrier()
        if CHART_IN:
            eb = e_ptr + t * 16
            for i in tl.static_range(4):
                q0 = tl.load(q_ptr + i * 3 + 0)
                q1 = tl.load(q_ptr + i * 3 + 1)
                q2 = tl.load(q_ptr + i * 3 + 2)
                t0 = _mhc_dot3(
                    q0, q1, q2, tl.load(zb + 0), tl.load(zb + 3), tl.load(zb + 6), False
                )
                t1 = _mhc_dot3(
                    q0, q1, q2, tl.load(zb + 1), tl.load(zb + 4), tl.load(zb + 7), False
                )
                t2 = _mhc_dot3(
                    q0, q1, q2, tl.load(zb + 2), tl.load(zb + 5), tl.load(zb + 8), False
                )
                for j in tl.static_range(4):
                    ev = _mhc_dot3(
                        t0,
                        t1,
                        t2,
                        tl.load(q_ptr + j * 3 + 0),
                        tl.load(q_ptr + j * 3 + 1),
                        tl.load(q_ptr + j * 3 + 2),
                        CHART_FMA2,
                    )
                    tl.store(eb + i * 4 + j, ev)
        h0 = tl.load(hb + 0)
        h1 = tl.load(hb + 1)
        h2 = tl.load(hb + 2)
        h3 = tl.load(hb + 3)
        if SPLIT:
            c_lo = tl.program_id(1) * BC
            c_hi = c_lo + BC
        else:
            c_lo = 0
            c_hi = C
        for c0 in range(c_lo, c_hi, BC):
            cols = c0 + tl.arange(0, BC)
            s0 = tl.load(v_base + 0 * C + cols).to(tl.float32)
            s1 = tl.load(v_base + 1 * C + cols).to(tl.float32)
            s2 = tl.load(v_base + 2 * C + cols).to(tl.float32)
            s3 = tl.load(v_base + 3 * C + cols).to(tl.float32)
            if AGG_TREE:
                p01 = tld.fma(h1, s1, tl.mul(h0, s0))
                p23 = tld.fma(h3, s3, tl.mul(h2, s2))
                aggv = tl.add(p01, p23)
            else:
                aggv = tld.fma(h3, s3, tld.fma(h2, s2, tld.fma(h1, s1, tl.mul(h0, s0))))
            tl.store(agg_ptr + t * C + cols, aggv.to(agg_ptr.dtype.element_ty))


def _torch_reduce_hw(T: int) -> tuple[int, int]:
    """(block_height, block_width) torch's reduce_kernel picks for a
    contiguous (T, nC>=2048) fp32 row reduce: H = min(last_pow2(T), 16),
    W = 512 // H (ReduceConfig::set_block_dimension, MAX_NUM_THREADS 512)."""
    h = 1
    while h * 2 <= T and h < 16:
        h *= 2
    return (h, 512 // h)


@functools.lru_cache(maxsize=None)
def _fp32_inv(nc: int) -> float:
    return float(
        torch.tensor(1.0, dtype=torch.float32)
        / torch.tensor(float(nc), dtype=torch.float32)
    )


if _TRAINER_FUSED:

    @triton.jit
    def _helmert_q(i: tl.constexpr, j: tl.constexpr):
        if j == 0:
            if i == 0:
                return 0.7071067811865476
            if i == 1:
                return -0.7071067811865476
            return 0.0
        if j == 1:
            if i == 0 or i == 1:
                return 0.408248290463863
            if i == 2:
                return -0.816496580927726
            return 0.0
        if i == 0 or i == 1 or i == 2:
            return 0.2886751345948129
        return -0.8660254037844386

    @triton.jit
    def _route_e(
        logits_ptr, row, row_mask, alpha_z, b_z_ptr, i: tl.constexpr, j: tl.constexpr
    ):
        value = row.to(tl.float32) * 0.0
        for a in tl.static_range(0, 3):
            for b in tl.static_range(0, 3):
                z = alpha_z * tl.load(
                    logits_ptr + row * 17 + 8 + a * 3 + b, mask=row_mask, other=0.0
                ).to(tl.float32) + tl.load(b_z_ptr + a * 3 + b).to(tl.float32)
                value += _helmert_q(i, a) * z * _helmert_q(j, b)
        return value

    @triton.jit
    def _route_forward_kernel(
        logits_ptr,
        alpha_pre_ptr,
        b_pre_ptr,
        alpha_post_ptr,
        b_post_ptr,
        alpha_z_ptr,
        b_z_ptr,
        eta_ptr,
        h_pre_ptr,
        h_post_ptr,
        h_res_ptr,
        rows,
        BLOCK_M: tl.constexpr,
    ):
        row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = row < rows
        alpha_pre = tl.load(alpha_pre_ptr).to(tl.float32)
        alpha_post = tl.load(alpha_post_ptr).to(tl.float32)
        alpha_z = tl.load(alpha_z_ptr).to(tl.float32)
        gamma = tl.sigmoid(tl.load(eta_ptr).to(tl.float32))
        for stream in tl.static_range(0, 4):
            pre_projection = tl.load(
                logits_ptr + row * 17 + stream, mask=row_mask, other=0.0
            ).to(tl.float32)
            post_projection = tl.load(
                logits_ptr + row * 17 + 4 + stream, mask=row_mask, other=0.0
            ).to(tl.float32)
            pre = tl.sigmoid(
                alpha_pre * pre_projection + tl.load(b_pre_ptr + stream).to(tl.float32)
            )
            post = 2.0 * tl.sigmoid(
                alpha_post * post_projection
                + tl.load(b_post_ptr + stream).to(tl.float32)
            )
            tl.store(h_pre_ptr + row * 4 + stream, pre, mask=row_mask)
            tl.store(h_post_ptr + row * 4 + stream, post, mask=row_mask)
        rho = row.to(tl.float32) * 0.0
        for out_stream in tl.static_range(0, 4):
            for in_stream in tl.static_range(0, 4):
                e = _route_e(
                    logits_ptr, row, row_mask, alpha_z, b_z_ptr, out_stream, in_stream
                )
                h0 = 0.625 if out_stream == in_stream else 0.125
                rho = tl.maximum(rho, -e / h0)
        rho = tl.maximum(rho, 0.0)
        denominator = 1.0 + rho
        for out_stream in tl.static_range(0, 4):
            for in_stream in tl.static_range(0, 4):
                e = _route_e(
                    logits_ptr, row, row_mask, alpha_z, b_z_ptr, out_stream, in_stream
                )
                h0 = 0.625 if out_stream == in_stream else 0.125
                identity = 1.0 if out_stream == in_stream else 0.0
                phi = h0 + e / denominator
                value = (1.0 - gamma) * identity + gamma * phi
                tl.store(
                    h_res_ptr + row * 16 + out_stream * 4 + in_stream,
                    value,
                    mask=row_mask,
                )

    @triton.jit
    def _normalize_project_fwd_kernel(
        x_ptr,
        w_ptr,
        logits_ptr,
        z_ptr,
        inv_ptr,
        rows,
        EPS: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        rows_idx = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols_idx = tl.arange(0, BLOCK_N)
        mask_m = rows_idx < rows
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        sumsq = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for start_k in tl.static_range(0, K, BLOCK_K):
            k_idx = start_k + tl.arange(0, BLOCK_K)
            mask_k = k_idx < K
            x_values = tl.load(
                x_ptr + rows_idx[:, None] * K + k_idx[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            w_values = tl.load(
                w_ptr + cols_idx[None, :] * K + k_idx[:, None],
                mask=(cols_idx[None, :] < N) & mask_k[:, None],
                other=0.0,
            )
            acc += tl.dot(x_values, w_values)
            x_fp32 = x_values.to(tl.float32)
            sumsq += tl.sum(x_fp32 * x_fp32, axis=1)
        inv = tl.rsqrt(sumsq / K + EPS)
        z_bf16 = acc.to(tl.bfloat16)
        output = z_bf16.to(tl.float32) * inv[:, None]
        offsets = rows_idx[:, None] * N + cols_idx[None, :]
        mask = mask_m[:, None] & (cols_idx[None, :] < N)
        tl.store(z_ptr + offsets, z_bf16, mask=mask)
        tl.store(logits_ptr + offsets, output, mask=mask)
        tl.store(inv_ptr + rows_idx, inv, mask=mask_m)

    @triton.jit
    def _aggregate_fwd_kernel(
        x_ptr,
        h_ptr,
        out_ptr,
        tokens: tl.constexpr,
        hidden: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = (rows[:, None] < tokens) & (cols[None, :] < hidden)
        x_base = rows[:, None] * (4 * hidden) + cols[None, :]
        h_base = rows * 4
        acc = tl.load(x_ptr + x_base, mask=mask, other=0.0).to(tl.float32)
        acc *= tl.load(h_ptr + h_base, mask=rows < tokens, other=0.0)[:, None]
        x1 = tl.load(x_ptr + x_base + hidden, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + x_base + 2 * hidden, mask=mask, other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + x_base + 3 * hidden, mask=mask, other=0.0).to(tl.float32)
        acc += x1 * tl.load(h_ptr + h_base + 1, mask=rows < tokens, other=0.0)[:, None]
        acc += x2 * tl.load(h_ptr + h_base + 2, mask=rows < tokens, other=0.0)[:, None]
        acc += x3 * tl.load(h_ptr + h_base + 3, mask=rows < tokens, other=0.0)[:, None]
        tl.store(out_ptr + rows[:, None] * hidden + cols[None, :], acc, mask=mask)

    @triton.jit
    def _mix_fwd_kernel(
        x_ptr,
        y_ptr,
        h_post_ptr,
        h_res_ptr,
        out_ptr,
        tokens: tl.constexpr,
        hidden: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = (rows[:, None] < tokens) & (cols[None, :] < hidden)
        x_base = rows[:, None] * (4 * hidden) + cols[None, :]
        y = tl.load(
            y_ptr + rows[:, None] * hidden + cols[None, :], mask=mask, other=0.0
        ).to(tl.float32)
        x0 = tl.load(x_ptr + x_base, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + x_base + hidden, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + x_base + 2 * hidden, mask=mask, other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + x_base + 3 * hidden, mask=mask, other=0.0).to(tl.float32)
        hp_base = rows * 4
        hr_base = rows * 16
        row_mask = rows < tokens
        for out_stream in tl.static_range(0, 4):
            coeff = hr_base + out_stream * 4
            acc = x0 * tl.load(h_res_ptr + coeff, mask=row_mask, other=0.0)[:, None]
            acc += (
                x1 * tl.load(h_res_ptr + coeff + 1, mask=row_mask, other=0.0)[:, None]
            )
            acc += (
                x2 * tl.load(h_res_ptr + coeff + 2, mask=row_mask, other=0.0)[:, None]
            )
            acc += (
                x3 * tl.load(h_res_ptr + coeff + 3, mask=row_mask, other=0.0)[:, None]
            )
            acc += (
                y
                * tl.load(h_post_ptr + hp_base + out_stream, mask=row_mask, other=0.0)[
                    :, None
                ]
            )
            out_offset = x_base + out_stream * hidden
            tl.store(out_ptr + out_offset, acc, mask=mask)

    @triton.jit
    def _neumaier_combine(left_sum, left_correction, right_sum, right_correction):
        total = left_sum + right_sum
        correction = tl.where(
            tl.abs(left_sum) >= tl.abs(right_sum),
            left_sum - total + right_sum,
            right_sum - total + left_sum,
        )
        return (total, left_correction + right_correction + correction)

    @triton.jit
    def _chunk_aggregate(
        x_ptr,
        x_base,
        hidden,
        offset,
        h0,
        h1,
        h2,
        h3,
        out_dtype: tl.constexpr,
        SIZE: tl.constexpr,
        STABLE: tl.constexpr,
    ):
        """Aggregate one channel chunk and take it through boundary 1.

        `hidden` is constexpr, so at widths where the two chunks sum to it
        exactly -- 2560 among them -- the mask is compile-time all-true and folds
        away. Measured: the backward's identical mask cost 0.2311 against 0.2318
        ms at 2560, inside the run-to-run spread.

        A masked lane reads 0, so its square contributes 0 to the row sum and the
        rstd is unaffected.
        """
        cols = offset + tl.arange(0, SIZE)
        keep = cols < hidden
        base = x_base + cols
        acc = h0 * tl.load(x_ptr + base, mask=keep, other=0.0).to(tl.float32)
        acc += h1 * tl.load(x_ptr + base + hidden, mask=keep, other=0.0).to(tl.float32)
        acc += h2 * tl.load(x_ptr + base + 2 * hidden, mask=keep, other=0.0).to(
            tl.float32
        )
        acc += h3 * tl.load(x_ptr + base + 3 * hidden, mask=keep, other=0.0).to(
            tl.float32
        )
        norm_input = acc.to(out_dtype).to(tl.float32)
        squares = norm_input * norm_input
        if STABLE:
            chunk_sum, chunk_correction = tl.reduce(
                (squares, tl.zeros_like(squares)), 0, _neumaier_combine
            )
        else:
            chunk_sum = tl.sum(squares, axis=0)
            chunk_correction = chunk_sum * 0.0
        return (norm_input, chunk_sum, chunk_correction)

    @triton.jit
    def _chunk_store(
        out_ptr,
        w_ptr,
        row,
        hidden,
        offset,
        norm_input,
        rstd,
        out_dtype: tl.constexpr,
        SIZE: tl.constexpr,
    ):
        """Normalize and apply the weight in fp32, rounding once at the store.

        This matches QuACK's RMSNorm, which the recipe selects with
        --kimi-rmsnorm-backend quack: "Keep reduction, normalization, and affine
        multiplication in FP32. Only the final output is rounded to BF16"
        (rmsnorm.py:341-350 loads to Float32, scales by rstd, multiplies the
        weight in Float32 and rounds at the store).

        It does NOT match eager KimiRMSNorm, which rounds between the two --
        `return self.weight * x.to(dtype)`. This kernel used to reproduce that
        rounding, because eager was what the branch norm ran; keeping it now
        would make the fused path disagree with the unfused one. Dropping it is
        also one conversion cheaper and one rounding more accurate.

        Boundary 1, the bf16 aggregate output, is unaffected: QuACK replaces the
        norm, not the aggregate.
        """
        cols = offset + tl.arange(0, SIZE)
        keep = cols < hidden
        xhat = norm_input * rstd
        weight = tl.load(w_ptr + cols, mask=keep, other=0.0).to(tl.float32)
        tl.store(
            out_ptr + row * hidden + cols, (xhat * weight).to(out_dtype), mask=keep
        )

    @triton.jit
    def _aggregate_norm_fwd_kernel(
        x_ptr,
        h_ptr,
        w_ptr,
        out_ptr,
        rstd_ptr,
        hidden: tl.constexpr,
        EPS: tl.constexpr,
        S0: tl.constexpr,
        S1: tl.constexpr,
        STABLE: tl.constexpr,
    ):
        row = tl.program_id(0)
        x_base = row * (4 * hidden)
        h0 = tl.load(h_ptr + row * 4 + 0).to(tl.float32)
        h1 = tl.load(h_ptr + row * 4 + 1).to(tl.float32)
        h2 = tl.load(h_ptr + row * 4 + 2).to(tl.float32)
        h3 = tl.load(h_ptr + row * 4 + 3).to(tl.float32)
        dt: tl.constexpr = out_ptr.dtype.element_ty
        a0, s, c = _chunk_aggregate(
            x_ptr, x_base, hidden, 0, h0, h1, h2, h3, dt, S0, STABLE
        )
        a1, s1, c1 = _chunk_aggregate(
            x_ptr, x_base, hidden, S0, h0, h1, h2, h3, dt, S1, STABLE
        )
        s, c = _neumaier_combine(s, c, s1, c1)
        rstd = tl.rsqrt((s + c) / hidden + EPS)
        tl.store(rstd_ptr + row, rstd)
        _chunk_store(out_ptr, w_ptr, row, hidden, 0, a0, rstd, dt, S0)
        _chunk_store(out_ptr, w_ptr, row, hidden, S0, a1, rstd, dt, S1)


def _aggnorm_chunks(hidden: int) -> tuple[int, int, bool]:
    """Exactly two power-of-two channel chunks, for ANY hidden.

    Used by both directions of the aggregate-norm, which is the point. The
    forward used to take up to five exact power-of-two chunks while the backward
    took two, and that asymmetry cost three things: widths needing six chunks
    were rejected outright (4032 among them), widths needing three to five
    reduced in a different order forward and backward so their rstd disagreed in
    the last bits, and worst, the forward ACCEPTED widths the backward refused --
    1728 built a model fine and died at the first backward.

    The rule: take the largest power of two at or below hidden, and round the
    remainder up to a power of two. Always two chunks, and EXACT whenever the
    remainder is itself a power of two:

        2560 -> (2048,  512)   sum 2560   exact, and identical to today's split
        1152 -> (1024,  128)   sum 1152   exact
        3072 -> (2048, 1024)   sum 3072   exact
        2048 -> (1024, 1024)   sum 2048   exact
        1728 -> (1024, 1024)   sum 2048   704 of the second chunk's 1024 lanes
                                          are live, 320 are masked, 15.6 percent
                                          of the total idle

    The production width is exact and gives (2048, 512), the same split both
    directions already used, so nothing about 2560 changes. Masking appears only
    at widths whose remainder is not a power of two, and there it costs idle
    lanes rather than correctness.

    What this gives up is the exact decomposition at widths needing three or more
    chunks -- 1728 was (1024, 512, 128, 64) and is now (1024, 1024) with 15.6
    percent of the lanes masked. That is not obviously a loss and has not been
    measured: at four warps a 64-wide chunk leaves half the 128 threads idle and
    a 128-wide chunk gives one element a thread, where both 1024-wide chunks give
    eight. Wide-and-masked may well beat narrow-and-exact. If it matters at some
    width later, measure it there rather than assume either way.

    Returns (S0, S1, masked).
    """
    if hidden <= 0:
        raise ValueError(f"hidden must be positive, got {hidden}")
    if hidden == 1:
        return (1, 1, True)
    top = 1 << hidden.bit_length() - 1
    if top == hidden:
        top >>= 1
    rest = triton.next_power_of_2(hidden - top) if triton is not None else 1
    while rest < hidden - top:
        rest <<= 1
    return (top, rest, top + rest != hidden)


class RaonMHC(nn.Module):
    """One mHC residual site: aggregate -> branch -> mix.

    ``site_index`` is the GLOBAL site index (2*layer for the attention site,
    2*layer+1 for the MLP site). It only sets ``b_pre``'s rotation at init, but
    a local index would give trained weights the wrong rotation on a partial
    load, so it is never derived locally.
    """

    _PHI_BLOCKS = ("phi_pre", "phi_post", "phi_z")
    _PLAIN = ("b_pre", "b_post", "b_z", "alpha_pre", "alpha_post", "alpha_z", "eta")

    def __init__(self, streams: int, hidden_size: int, site_index: int) -> None:
        super().__init__()
        n = self.streams = streams
        nz = (n - 1) ** 2
        self._rows = {
            "phi_pre": slice(0, n),
            "phi_post": slice(n, 2 * n),
            "phi_z": slice(2 * n, 2 * n + nz),
        }
        self.phi = nn.Parameter(torch.zeros(2 * n + nz, n * hidden_size))
        sig_off = min(0.1, 0.5 / (n - 1))
        logit = lambda p: math.log(p / (1.0 - p))
        b_pre = torch.full((n,), logit(sig_off))
        b_pre[site_index % n] = logit(1.0 - (n - 1) * sig_off)
        self.b_pre = nn.Parameter(b_pre)
        self.b_post = nn.Parameter(torch.zeros(n))
        self.b_z = nn.Parameter(torch.zeros(nz))
        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_z = nn.Parameter(torch.tensor(0.01))
        gamma0 = raon_mhc_gamma_init(n)
        self.eta = nn.Parameter(
            torch.tensor(math.log(gamma0 / (1.0 - gamma0)), dtype=torch.float32)
        )
        eye = torch.eye(n, dtype=torch.float32)
        self.register_buffer("Q", helmert_basis(n), persistent=False)
        self.register_buffer(
            "H0",
            0.5 * (eye + torch.full((n, n), 1.0 / n, dtype=torch.float32)),
            persistent=False,
        )
        self.register_buffer("eye", eye, persistent=False)

    def load_weight(self, suffix: str, w: Tensor) -> None:
        """Load one checkpoint tensor named ``...{self_attn,mlp}_mhc.<suffix>``.

        Megatron nests the chart params under a ``res`` submodule and keeps the
        three phi blocks apart; this port is flat and fuses phi into one GEMM,
        so both spellings route here.
        """
        suffix = suffix.removeprefix("res.")
        if suffix in self._rows:
            self.phi.data[self._rows[suffix]].copy_(w)
        elif suffix in self._PLAIN:
            getattr(self, suffix).data.copy_(w)
        else:
            raise KeyError(f"unknown mHC parameter {suffix!r}")

    def _refit_set_buf(self, name, value):
        """Update a buffer in place. Captured decode CUDA graphs bake in
        buffer pointers, and a refit re-runs load_weights -> finalize; if the
        buffers were re-registered (new storage), graph replay would keep
        reading the stale buffers and produce NaN. When shape/dtype/device
        match, copy_ into the existing storage (pointer unchanged); only
        first creation registers a new buffer. inference_mode is needed
        because the refit comes from a worker extension outside
        InferenceMode."""
        old = getattr(self, name, None)
        if (
            isinstance(old, torch.Tensor)
            and old.shape == value.shape
            and (old.dtype == value.dtype)
            and (old.device == value.device)
        ):
            with torch.inference_mode():
                old.copy_(value)
            return
        if old is not None:
            raise RuntimeError(
                f"mHC refit contract violation: buffer {name!r} would need re-registration after a structure change (old {tuple(getattr(old, 'shape', ()))}/{getattr(old, 'dtype', None)}/{getattr(old, 'device', None)} -> new {tuple(value.shape)}/{value.dtype}/{value.device}). Re-registering leaves stale pointers in captured decode graphs and makes every request return NaN; a graph re-capture path is required first."
            )
        self.register_buffer(name, value, persistent=False)

    def finalize(self) -> None:
        """Precompute forward-time constants from the loaded weights.

        Pure re-layout of trained parameters; the forward that consumes these
        buffers is bit-exact vs the per-slice math (same values, same op
        order). Called from Raon1MiniForCausalLM.load_weights; the forward has
        a lazy fallback for bare-module tests.
        """
        n, nz = (self.streams, (self.streams - 1) ** 2)
        a = torch.cat(
            [
                self.alpha_pre.float().expand(n),
                self.alpha_post.float().expand(n),
                self.alpha_z.float().expand(nz),
            ]
        )
        b = torch.cat([self.b_pre.float(), self.b_post.float(), self.b_z.float()])
        g = torch.sigmoid(self.eta.float())
        self._refit_set_buf("A_row", a.detach().clone())
        self._refit_set_buf("B_row", b.detach().clone())
        self._refit_set_buf("g_eye", ((1.0 - g) * self.eye).detach())
        self._refit_set_buf("g_gate", g.detach().clone())
        self._refit_set_buf("negH0", (-self.H0).detach())
        if _PHI_DIRECT and _PHI_DIRECT_MODE != "m32":
            _phi_all_finalize(self, 2 * n + nz)
        elif _PHI_DIRECT:
            R = 2 * n + nz
            if _PHI_DIRECT_N_PAD <= R:
                raise RuntimeError(
                    f"{_PHI_DIRECT_ENV}: pad {_PHI_DIRECT_N_PAD} does not exceed the real row count {R}; the pad would truncate phi."
                )
            pp = torch.zeros(
                _PHI_DIRECT_N_PAD,
                self.phi.shape[1],
                dtype=self.phi.dtype,
                device=self.phi.device,
            )
            pp[:R] = self.phi.detach()
            self._refit_set_buf("phi_padded", pp)
            self._phi_direct_tactic = _phi_default_tactic(
                _PHI_DIRECT_M, _PHI_DIRECT_N_PAD, self.phi.shape[1]
            )
            if pp.is_cuda:
                with torch.no_grad():
                    zin = torch.zeros(
                        _PHI_DIRECT_M,
                        self.phi.shape[1],
                        dtype=pp.dtype,
                        device=pp.device,
                    )
                    zout = torch.empty(
                        _PHI_DIRECT_M,
                        _PHI_DIRECT_N_PAD,
                        dtype=pp.dtype,
                        device=pp.device,
                    )
                    _phi_run_direct_dense(
                        zin, pp.t(), zout, pdl=False, tactic=self._phi_direct_tactic
                    )
                print(
                    f"[va] {_PHI_DIRECT_ENV}=1 armed: phi_padded [{_PHI_DIRECT_N_PAD}, {self.phi.shape[1]}] built, tactic={self._phi_direct_tactic}, direct kernel precompiled (finalize).",
                    flush=True,
                )

    def _forward_trainer_fused(
        self,
        s: Tensor,
        branch: Callable[[Tensor], Tensor],
        branch_norm_weight: Tensor | None = None,
        branch_norm_eps: float | None = None,
    ) -> Tensor:
        """The trainer's fused realization, launched exactly as the trainer
        launches it (fused_raon_mhc.py `_NormalizeProject`/`_RaonRouteN4`/
        `_AggregateN4`/`_MixN4` forward wrappers): same kernels, same launch
        schedules, forward only. Raw checkpoint parameters are consumed (not
        the finalize() re-layout) because that is what the trainer reads.

        ``branch_norm_weight`` (RAON_MHC_AGGNORM_FUSED, wired by model.py):
        replace the plain aggregate with the trainer's fused
        `_aggregate_norm_fwd_kernel` (`_AggregateNormN4.forward` launch), so
        ``branch`` receives norm(H_pre X) computed in-kernel from the fp32
        aggregate accumulator and must NOT normalize again — exactly the
        trainer's `raon_mhc_aggregate_norm` contract (mhc_residual.py)."""
        T, n, C = s.shape
        assert s.dtype == torch.bfloat16 and self.phi.dtype == torch.bfloat16
        v = s.view(T, n * C)
        rows = T
        _sk = (rows, C, s.device, s.dtype)
        _sc = getattr(self, "_mhc_scratch", None)
        if _sc is None or _sc["key"] != _sk:
            _sc = {
                "key": _sk,
                "logits": torch.empty((rows, 17), dtype=torch.float32, device=s.device),
                "z_unscaled": torch.empty(
                    (rows, 17), dtype=torch.bfloat16, device=s.device
                ),
                "inv": torch.empty(rows, dtype=torch.float32, device=s.device),
                "h_pre": torch.empty((rows, 4), dtype=torch.float32, device=s.device),
                "h_post": torch.empty((rows, 4), dtype=torch.float32, device=s.device),
                "h_res": torch.empty(
                    (rows, 4, 4), dtype=torch.float32, device=s.device
                ),
                "agg": torch.empty((rows, C), dtype=s.dtype, device=s.device),
                "rstd": torch.empty(rows, dtype=torch.float32, device=s.device),
                "chunks": _aggnorm_chunks(C)[:2],
                "mix": (512, 2)
                if C % 512 == 0
                else (256, 8)
                if C >= 1536
                else (128, 4),
                "agg_block_c": 512 if C % 512 == 0 else 128,
                "bnw_checked": None,
            }
            self._mhc_scratch = _sc
        logits = _sc["logits"]
        z_unscaled = _sc["z_unscaled"]
        inv = _sc["inv"]
        _normalize_project_fwd_kernel[triton.cdiv(rows, 16),](
            v,
            self.phi,
            logits,
            z_unscaled,
            inv,
            rows,
            EPS=_EPS,
            K=n * C,
            N=17,
            BLOCK_M=16,
            BLOCK_N=32,
            BLOCK_K=128,
            num_warps=8,
            num_stages=2,
            enable_fp_fusion=False,
        )
        h_pre, h_post, h_res = (_sc["h_pre"], _sc["h_post"], _sc["h_res"])
        _route_forward_kernel[triton.cdiv(rows, 128),](
            logits,
            self.alpha_pre,
            self.b_pre,
            self.alpha_post,
            self.b_post,
            self.alpha_z,
            self.b_z,
            self.eta,
            h_pre,
            h_post,
            h_res,
            rows,
            BLOCK_M=128,
            num_warps=2,
            enable_fp_fusion=False,
        )
        agg = _sc["agg"]
        if branch_norm_weight is not None:
            if not _AGGNORM_FUSED:
                raise RuntimeError(
                    "branch_norm_weight passed with RAON_MHC_AGGNORM_FUSED=0"
                )
            _bid = (id(branch_norm_weight), branch_norm_weight.data_ptr())
            if _sc["bnw_checked"] != _bid:
                if branch_norm_weight.shape != (C,):
                    raise ValueError(
                        f"branch norm weight must be ({C},), got {tuple(branch_norm_weight.shape)}"
                    )
                if not branch_norm_weight.is_contiguous():
                    raise ValueError("branch norm weight must be contiguous")
                _sc["bnw_checked"] = _bid
            chunks = _sc["chunks"]
            rstd = _sc["rstd"]
            _aggregate_norm_fwd_kernel[rows,](
                s,
                h_pre,
                branch_norm_weight,
                agg,
                rstd,
                hidden=C,
                EPS=float(branch_norm_eps),
                S0=chunks[0],
                S1=chunks[1],
                STABLE=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            agg_block_c = _sc["agg_block_c"]
            _aggregate_fwd_kernel[triton.cdiv(rows, 2), triton.cdiv(C, agg_block_c)](
                s,
                h_pre,
                agg,
                tokens=rows,
                hidden=C,
                BLOCK_M=2,
                BLOCK_C=agg_block_c,
                num_warps=4,
                enable_fp_fusion=False,
            )
        y = branch(agg)
        y_flat = y.reshape(rows, C).to(torch.bfloat16).contiguous()
        mix_block_c, mix_warps = _sc["mix"]
        out = torch.empty_like(s)
        _mix_fwd_kernel[rows, triton.cdiv(C, mix_block_c)](
            s,
            y_flat,
            h_post,
            h_res,
            out,
            tokens=rows,
            hidden=C,
            BLOCK_M=1,
            BLOCK_C=mix_block_c,
            num_warps=mix_warps,
            enable_fp_fusion=False,
        )
        return out

    def _forward_fused(self, s: Tensor, branch: Callable[[Tensor], Tensor]) -> Tensor:
        """~9 launches per site (vs 29 on the torch path): cublas logits,
        square, torch mean, head epilogue, torch chart einsum, mix."""
        T, n, C = s.shape
        v = s.view(T, n * C)
        R = 2 * n + (n - 1) ** 2
        if _BATCH_INVARIANT:
            mm_out = matmul_persistent(v, self.phi.t())
        elif _PHI_DIRECT:
            _phi_direct_self_check(self, v)
            mm_out = _phi_direct_mm(self, v)
        else:
            mm_out = torch.mm(v, self.phi.t())
        mm_stride = mm_out.shape[1]
        if _HEAD_RMS or _HEAD_CHART or _SPLIT_MAX_T > 0 or _STAGES1:
            return self._forward_fused_sgl(s, branch, mm_out)
        sq = torch.empty(T, n * C, device=s.device, dtype=torch.float32)
        total = T * n * C
        _mhc_square[triton.cdiv(total, 1024),](
            s, sq, total, B=1024, enable_fp_fusion=False
        )
        mean = sq.mean(-1)
        head = torch.empty(T, 2 * n, device=s.device, dtype=torch.float32)
        z = torch.empty(T, n - 1, n - 1, device=s.device, dtype=torch.float32)
        agg = torch.empty(T, C, device=s.device, dtype=s.dtype)
        agg_tree = _FIXED_AGG_TREE if _BATCH_INVARIANT else T <= 8
        _mhc_site_head[T,](
            s,
            mm_out,
            mean,
            self.A_row,
            self.B_row,
            head,
            z,
            agg,
            NC=n * C,
            C=C,
            N=n,
            R=R,
            MMS=mm_stride,
            BC=512,
            EPS=_EPS,
            AGG_TREE=agg_tree,
            num_warps=4,
            enable_fp_fusion=False,
        )
        e = torch.einsum("im,bmk,jk->bij", self.Q, z, self.Q).contiguous()
        y = branch(agg)
        out = torch.empty_like(s)
        mix_one = _FIXED_MIX_ONE if _BATCH_INVARIANT else T == 1
        _launch_site_mix(
            (T, C // 512),
            s,
            y,
            head,
            e,
            self.H0,
            self.negH0,
            self.g_eye,
            self.g_gate,
            out,
            C=C,
            N=n,
            BC=512,
            MIX_ONE=mix_one,
            num_warps=4,
        )
        return out

    def _forward_fused_sgl(
        self, s: Tensor, branch: Callable[[Tensor], Tensor], mm_out: Tensor
    ) -> Tensor:
        """_forward_fused with the SGLang-port levers (RAON_MHC_STAGES1 /
        HEAD_RMS / HEAD_CHART / SPLIT_MAX_T). Same values as the base site
        bit-for-bit; only where the rms mean and the chart are computed and
        how the head/mix grids are cut change. Down to 3 launches per site
        (phi GEMM, head, mix) with HEAD_RMS + HEAD_CHART."""
        global _SGL_ANNOUNCED
        T, n, C = s.shape
        NC = n * C
        R = 2 * n + (n - 1) ** 2
        ks = {"num_stages": 1} if _STAGES1 else {}
        rms_in = _HEAD_RMS and NC % 2048 == 0
        chart_in = _HEAD_CHART and self.Q.dtype == torch.float32
        split = _SPLIT_MAX_T > 0 and T <= _SPLIT_MAX_T
        if rms_in:
            mean = mm_out
        else:
            sq = torch.empty(T, NC, device=s.device, dtype=torch.float32)
            total = T * NC
            _mhc_square[triton.cdiv(total, 1024),](
                s, sq, total, B=1024, **ks, enable_fp_fusion=False
            )
            mean = sq.mean(-1)
        head = torch.empty(T, 2 * n, device=s.device, dtype=torch.float32)
        z = torch.empty(T, n - 1, n - 1, device=s.device, dtype=torch.float32)
        e = (
            torch.empty(T, n, n, device=s.device, dtype=torch.float32)
            if chart_in
            else z
        )
        agg = torch.empty(T, C, device=s.device, dtype=s.dtype)
        agg_tree = _FIXED_AGG_TREE if _BATCH_INVARIANT else T <= 8
        rh, rw = _torch_reduce_hw(T)
        hbc = _SPLIT_HEAD_BC if split and C % _SPLIT_HEAD_BC == 0 else 512
        _mhc_site_head_v2[(T, C // hbc) if split else (T,)](
            s,
            mm_out,
            mean,
            self.A_row,
            self.B_row,
            head,
            z,
            e,
            self.Q,
            agg,
            NC=NC,
            C=C,
            N=n,
            R=R,
            MMS=mm_out.shape[1],
            BC=hbc,
            EPS=_EPS,
            AGG_TREE=agg_tree,
            RMS_IN=rms_in,
            RH=rh,
            RW=rw,
            FACTOR=_fp32_inv(NC),
            CHART_IN=chart_in,
            CHART_FMA2=T >= 5,
            SPLIT=split,
            num_warps=4,
            **ks,
            enable_fp_fusion=False,
        )
        _SGL_STATS["head_v2"] += 1
        _SGL_STATS["rms_in"] += int(rms_in)
        _SGL_STATS["chart_in"] += int(chart_in)
        _SGL_STATS["split"] += int(split)
        if not _SGL_ANNOUNCED:
            _SGL_ANNOUNCED = True
            print(
                f"[mhc-sglang] armed: STAGES1={_STAGES1} HEAD_RMS={_HEAD_RMS}(in-kernel={rms_in}) HEAD_CHART={_HEAD_CHART}(in-kernel={chart_in}) SPLIT_MAX_T={_SPLIT_MAX_T} SPLIT_MIX_BC={_SPLIT_MIX_BC} SPLIT_HEAD_BC={_SPLIT_HEAD_BC} first T={T}",
                flush=True,
            )
        if not chart_in:
            e = torch.einsum("im,bmk,jk->bij", self.Q, z, self.Q).contiguous()
        y = branch(agg)
        out = torch.empty_like(s)
        mix_one = _FIXED_MIX_ONE if _BATCH_INVARIANT else T == 1
        mbc = _SPLIT_MIX_BC if split and C % _SPLIT_MIX_BC == 0 else 512
        _launch_site_mix(
            (T, C // mbc),
            s,
            y,
            head,
            e,
            self.H0,
            self.negH0,
            self.g_eye,
            self.g_gate,
            out,
            C=C,
            N=n,
            BC=mbc,
            MIX_ONE=mix_one,
            num_warps=4,
            **ks,
        )
        return out

    def forward(
        self,
        s: Tensor,
        branch: Callable[[Tensor], Tensor],
        branch_norm_weight: Tensor | None = None,
        branch_norm_eps: float | None = None,
    ) -> Tensor:
        """s: (T, n, C) streams. branch: (T, C) -> (T, C), no residual inside.

        ``branch_norm_weight``/``branch_norm_eps`` exist only for
        RAON_MHC_AGGNORM_FUSED (model.py passes them when the lever is on, and
        then ``branch`` must not pre-normalize — the site does it in the fused
        aggregate). Only the trainer-fused path realizes them; reaching any
        other path with a weight set would silently skip the branch norm, so
        it raises instead."""
        n, dt = (self.streams, s.dtype)
        if not hasattr(self, "A_row"):
            self.finalize()
        if (
            s.is_cuda
            and n == 4
            and (s.shape[-1] % 16 == 0)
            and (_TRAINER_FUSED_ENV or raon_phase.aligned("mhc.site"))
        ):
            return self._forward_trainer_fused(
                s.contiguous(), branch, branch_norm_weight, branch_norm_eps
            )
        if branch_norm_weight is not None:
            raise RuntimeError(
                "branch_norm_weight given but the trainer-fused site path is unavailable (RAON_MHC_AGGNORM_FUSED requires the RAON_MHC_TRAINER_FUSED CUDA path); refusing to run with the branch pre-norm silently dropped"
            )
        if _FUSED and s.is_cuda and (n == 4) and (s.shape[-1] % 512 == 0):
            if not _CHECK:
                return self._forward_fused(s.contiguous(), branch)
            ys: list[Tensor] = []

            def tap(a: Tensor) -> Tensor:
                y = branch(a)
                ys.append((a, y))
                return y

            out_f = self._forward_fused(s.contiguous(), tap)
            out_t = self._forward_torch(s, lambda a: ys[0][1])
            agg_f = ys[0][0]
            v0 = s.reshape(s.shape[0], -1)
            inv0 = torch.rsqrt(v0.float().pow(2).mean(-1, keepdim=True) + _EPS)
            pre0 = self.A_row * (torch.mm(v0, self.phi.t()) * inv0) + self.B_row
            sig0 = torch.sigmoid(pre0[:, : 2 * n])
            agg_t = torch.einsum("mn,mnc->mc", sig0[:, :n], s.float()).to(s.dtype)
            na = (agg_f.view(torch.int16) != agg_t.view(torch.int16)).sum().item()
            no = (out_f.view(torch.int16) != out_t.view(torch.int16)).sum().item()
            if na or no:
                import sys

                print(
                    f"MHC_CHECK MISMATCH T={s.shape[0]} agg_neq={na} out_neq={no}",
                    file=sys.stderr,
                    flush=True,
                )
            return out_t
        return self._forward_torch(s, branch)

    def _forward_torch(self, s: Tensor, branch: Callable[[Tensor], Tensor]) -> Tensor:
        n, dt = (self.streams, s.dtype)
        v = s.reshape(s.shape[0], -1)
        inv = torch.rsqrt(v.float().pow(2).mean(-1, keepdim=True) + _EPS)
        if _BATCH_INVARIANT and v.is_cuda:
            mm = matmul_persistent(v, self.phi.t())
        elif _PHI_DIRECT and v.is_cuda:
            _phi_direct_self_check(self, v)
            mm = _phi_direct_mm(self, v)[:, : self.phi.shape[0]]
        else:
            mm = torch.mm(v, self.phi.t())
        logits = mm * inv
        pre = self.A_row * logits + self.B_row
        sig = torch.sigmoid(pre[:, : 2 * n])
        h_pre = sig[:, :n]
        h_post = 2.0 * sig[:, n : 2 * n]
        z = pre[:, 2 * n :].reshape(-1, n - 1, n - 1)
        e = torch.einsum("im,bmk,jk->bij", self.Q, z, self.Q)
        rho = (e / self.negH0).amax(dim=(-2, -1)).clamp_min(0.0)
        phi = self.H0 + e / (1.0 + rho)[:, None, None]
        h_res = self.g_eye + self.g_gate * phi
        sf = s.float()
        y = branch(torch.einsum("mn,mnc->mc", h_pre, sf).to(dt))
        return (
            torch.einsum("mij,mjc->mic", h_res, sf)
            + h_post.unsqueeze(-1) * y.unsqueeze(1)
        ).to(dt)
