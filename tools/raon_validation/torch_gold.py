# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent FP32 Torch checks of real TP1 Raon RDA/MLA decode inputs.

Each comparison starts from the native operation's current input/cache state.
It checks local arithmetic, not the correctness of the scheduler's history.
"""

import functools
import inspect

import torch

from .common import error, write_json
from .model import cpu
from .reference import rda


class TorchGold:
    def __init__(self, guard):
        assert not guard.args.common_rda, "Gold checks require production RDA paths"
        assert guard.args.mla_backend == "ROCM_AITER_MLA"
        self.guard = guard
        self.layer = None
        self.raw_value = None
        self.delta = None
        self.cases = []
        self.counts = {}
        self.failed_dumps = set()

    def select(self, part, tokens):
        if not self.guard.active or not 0 < tokens <= 8:
            return False
        key = part + ":" + self.guard.current_layer
        n = self.counts.get(key, 0) + 1
        self.counts[key] = n
        return n in [1, 2, 4, 8, 16]

    def record(self, case, tensors):
        self.cases.append(case)
        failed = any(
            not v["pass"] for v in case.values() if isinstance(v, dict) and "pass" in v
        )
        part = case["part"]
        if failed and part not in self.failed_dumps:
            torch.save(cpu(tensors), self.guard.args.output + ".gold-" + part + ".pt")
            self.failed_dumps.add(part)
        write_json(self.guard.args.output + ".gold.json", self.summary())

    def summary(self):
        return {
            "scope": (
                "Native operation versus independent FP32 Torch from identical "
                "live inputs/state; TP1 single-request decode only; RDA "
                "post-convolution core and MLA latent attention before projection"
            ),
            "rda_references": {
                "rda_equations": (
                    "Separate erase/write sigmoids on original post-conv V; "
                    "includes value-fold rounding error"
                ),
                "working_input_equations": (
                    "Same stored BF16 folded V as native; isolates kernel arithmetic"
                ),
            },
            "mla_reference": (
                "Dense causal FP32 attention of exact stored/dequantized Q and KV; "
                "no FP8 probability rounding in gold"
            ),
            "counts": self.counts,
            "cases": self.cases,
        }

    def install(self, kda):
        ratio_fn = kda.compute_rda_value_ratio
        scale_fn = kda.RaonKDAAttention._scale_packed_v

        @functools.wraps(ratio_fn)
        def ratio(beta, delta):
            if self.guard.active:
                self.delta = delta.detach().clone()
            return ratio_fn(beta, delta)

        @functools.wraps(scale_fn)
        def scale(layer, mixed, ratio, beta):
            if self.guard.active and mixed.shape[0] <= 8:
                self.raw_value = mixed[:, 2 * layer.local_projection_size :].clone()
            return scale_fn(layer, mixed, ratio, beta)

        kda.compute_rda_value_ratio = ratio
        kda.RaonKDAAttention._scale_packed_v = scale
        for name in ["fused_recurrent_kda", "fused_recurrent_kda_packed_decode"]:
            original = getattr(kda, name)
            signature = inspect.signature(original)

            def wrapper(*args, _fn=original, _sig=signature, **kwargs):
                bound = _sig.bind(*args, **kwargs).arguments
                packed = "mixed_qkv" in bound
                tokens = bound["mixed_qkv"].shape[0] if packed else bound["q"].shape[1]
                if not self.select("rda", tokens):
                    return _fn(*args, **kwargs)
                indices = (
                    bound["state_indices"] if packed else bound["ssm_state_indices"]
                )
                # A single request is used to avoid guessing mixed-batch mappings.
                if (packed and tokens != 1) or (not packed and indices.shape[0] != 1):
                    return _fn(*args, **kwargs)
                if packed:
                    slot = int(indices[0])
                    h, d = bound["raw_g"].shape[2:]
                    q, k, v = [
                        x.reshape(1, tokens, h, d).clone()
                        for x in bound["mixed_qkv"].split(h * d, -1)
                    ]
                    written = indices[:1].long()
                else:
                    accepted = bound.get("num_accepted_tokens")
                    slot = int(
                        indices[0, int(accepted[0]) - 1 if accepted is not None else 0]
                    )
                    q, k, v = [bound[x].clone() for x in ["q", "k", "v"]]
                    written = indices[0, :tokens].long()
                assert slot > 0 and bool((written > 0).all())
                state = bound["initial_state"][slot : slot + 1].clone()
                assert self.raw_value is not None and self.delta is not None
                raw_v = self.raw_value.reshape_as(v).clone()
                delta = self.delta[:, :tokens].clone()
                # Core equations use the original V and independently activated
                # write/erase gates, without calling native ratio/gate helpers.
                common = [bound["raw_g"], bound["raw_beta"]]
                tail = [bound["A_log"], bound["dt_bias"], state, bound["lower_bound"]]
                ideal, ideal_states = rda(q, k, raw_v, *common, delta, *tail)
                working, working_states = rda(
                    q, k, v, *common, torch.zeros_like(delta), *tail
                )
                out = _fn(*args, **kwargs)
                actual = out[0]
                final = bound["initial_state"][written]
                case = {
                    "part": "rda",
                    "layer": self.guard.current_layer,
                    "kernel": "packed_AR" if packed else "recurrent_SD",
                    "tokens": tokens,
                    "rda_equations": error(actual, ideal),
                    "rda_states": error(final, ideal_states[0]),
                    "working_input_equations": error(actual, working),
                    "working_input_states": error(final, working_states[0]),
                }
                self.record(
                    case,
                    dict(
                        case=case,
                        q=q,
                        k=k,
                        original_v=raw_v,
                        folded_v=v,
                        delta=delta,
                        raw_g=bound["raw_g"],
                        raw_beta=bound["raw_beta"],
                        A_log=bound["A_log"],
                        bias=bound["dt_bias"],
                        initial=state,
                        native_output=actual,
                        gold=ideal,
                        gold_working=working,
                    ),
                )
                return out

            setattr(kda, name, wrapper)

    def mla(self, bound):
        q = bound["q"]
        if not self.select("mla", q.shape[0]):
            return
        qo, indptr = bound["qo_indptr"], bound["kv_indptr"]
        if qo.numel() != 2:
            return
        length = int(indptr[-1])
        idx = bound["kv_indices"][:length].long()
        cache = bound["kv_buffer"]
        kv = cache.reshape(-1, cache.shape[-1])[idx]
        qf = q[:, :40].float() * bound.get("q_scale", 1.0)
        kf = kv.float() * bound.get("kv_scale", 1.0)
        scores = torch.einsum("qhd,kd->qhk", qf, kf) * bound["sm_scale"]
        if bound.get("causal", True):
            mask = torch.arange(length, device=q.device)[None, :] > (
                length - q.shape[0] + torch.arange(q.shape[0], device=q.device)[:, None]
            )
            scores.masked_fill_(mask[:, None, :], -torch.inf)
        expected = torch.einsum("qhk,kd->qhd", scores.softmax(-1), kf[:, :512])
        actual = bound["o"][:, :40]
        case = {
            "part": "mla",
            "layer": self.guard.current_layer,
            "tokens": q.shape[0],
            "kv_length": length,
            "dense_fp32": error(actual, expected, 0.02, 0.05),
        }
        self.record(
            case,
            dict(
                case=case,
                q=q[:, :40],
                kv=kv,
                q_scale=bound.get("q_scale", 1.0),
                kv_scale=bound.get("kv_scale", 1.0),
                sm_scale=bound["sm_scale"],
                native_output=actual,
                gold=expected,
            ),
        )
