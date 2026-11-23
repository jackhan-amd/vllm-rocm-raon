# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct FP32 RDA equations, with separate erase and write gates."""

import torch
import torch.nn.functional as F


def rda(q, k, v, raw_g, beta, delta, a_log, bias, initial, lower_bound=-5.0):
    # Layouts: B,T,H,D and B,H,V,K. No production gate/recurrence helpers.
    q, k, v = q.float(), k.float(), v.float()
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    h, d = q.shape[-2:]
    logits = raw_g.float() + bias.reshape(h, d)
    if lower_bound is None:
        gate = -a_log.exp()[None, None, :, None] * F.softplus(logits)
    else:
        gate = lower_bound * torch.sigmoid(a_log.exp()[None, None, :, None] * logits)
    decay = gate.exp()
    erase, write = beta.float().sigmoid(), (beta.float() + delta.float()).sigmoid()
    state = initial.float().clone()
    outputs, states = [], []
    for t in range(q.shape[1]):
        state = state * decay[:, t, :, None, :]
        predicted = (state * k[:, t, :, None, :]).sum(-1)
        update = write[:, t, :, None] * v[:, t] - erase[:, t, :, None] * predicted
        state = state + update[..., None] * k[:, t, :, None, :]
        outputs.append((state * q[:, t, :, None, :]).sum(-1) * d**-0.5)
        states.append(state.clone())
    return torch.stack(outputs, 1), torch.stack(states, 1)


def convolution(raw, history, weight):
    """FP32 depthwise causal convolution and SiLU, rounded at the BF16 boundary."""
    joined = torch.cat([history.float(), raw.float().transpose(1, 2)], -1)
    out = torch.zeros_like(raw.transpose(1, 2), dtype=torch.float32)
    for j in range(weight.shape[-1]):
        out += (
            joined[:, :, j : j + raw.shape[1]] * weight[None, :, j].float()[..., None]
        )
    return F.silu(out).to(raw.dtype).transpose(1, 2).contiguous(), joined
