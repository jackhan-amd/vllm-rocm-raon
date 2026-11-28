/* SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 * Experimental one-CTA/head SD: conv, recurrent checkpoints and gated norm.
 */
#define RAON_RDA_NO_MODULE
#include "rocm_hip_rda_ar.hip"

namespace {
constexpr int H = 40, D = 128, P = H * D;

__device__ __forceinline__ float conv4(bf16_t h0, bf16_t h1, bf16_t h2,
                                       bf16_t x, const float* weight) {
  float acc = 0.f;
  acc += __bfloat162float(h0) * weight[0];
  acc += __bfloat162float(h1) * weight[P];
  acc += __bfloat162float(h2) * weight[2 * P];
  acc += __bfloat162float(x) * weight[3 * P];
  return __bfloat162float(bf16_store(silu_fast(acc)));
}

__global__ __launch_bounds__(256) void sd_conv_recurrence_norm(
    const bf16_t* x, const float* weights, bf16_t* conv, const bf16_t* g,
    const bf16_t* beta, const float* ratio, const float* a_log,
    const float* bias, const int* idx, const int* accepted, const int* cu,
    float* state, const bf16_t* gate, const float* norm, bf16_t* output,
    int steps, int slots, int64_t xr, int64_t cs, int64_t cc, int64_t ct,
    int64_t ss, int64_t br, int64_t rr, int64_t gr, float bound, float eps) {
  const int n = blockIdx.y, head = blockIdx.x, tid = threadIdx.x;
  const int lo = cu[n], hi = cu[n + 1], a = accepted[n];
  const int span = hi - lo;
  // Every packed output position is initialized, including a NULL suffix.
  if (tid < D) {
    for (int t = 0; t < span; ++t)
      output[int64_t(lo + t) * P + head * D + tid] = bf16_store(0.f);
  }
  if (a < 1 || a > steps) return;
  const int first = idx[n * steps], read = idx[n * steps + a - 1];
  if (first <= 0 || first >= slots || read <= 0 || read >= slots) return;
  int count = min(span, steps);
  for (int t = 0; t < count; ++t) {
    const int write = idx[n * steps + t];
    if (write <= 0 || write >= slots) {
      count = t;
      break;
    }
  }
  if (!count) return;

  const int lane = tid & 15, grp = tid / 16;
  f32x4 h[8][2];
#pragma unroll
  for (int row = 0; row < 8; ++row) {
#pragma unroll
    for (int j = 0; j < 2; ++j)
      h[row][j] = load_state(state + int64_t(read) * ss + head * D * D +
                             (row * 16 + grp) * D + 4 * lane + j * 64);
  }
  bf16_t qh[3], kh[3], vh[3];
  bf16_t *qc = nullptr, *kc = nullptr, *vc = nullptr;
  if (tid < D) {
    qc = conv + int64_t(first) * cs + (head * D + tid) * cc;
    kc = qc + P * cc;
    vc = kc + P * cc;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      qh[j] = qc[(a - 1 + j) * ct];
      kh[j] = kc[(a - 1 + j) * ct];
      vh[j] = vc[(a - 1 + j) * ct];
    }
    qc[0] = qh[1];
    qc[ct] = qh[2];
    kc[0] = kh[1];
    kc[ct] = kh[2];
    vc[0] = vh[1];
    vc[ct] = vh[2];
  }
  __shared__ float q[D], k[D], decay[D], v[D], values[D];
  __shared__ float scratch[kPartialBase + kRows];
  for (int token = 0; token < count; ++token) {
    const int pos = lo + token, write = idx[n * steps + token];
    float qv = 0.f, kv = 0.f;
    if (tid < D) {
      const int channel = head * D + tid;
      const bf16_t qx = x[int64_t(pos) * xr + channel];
      const bf16_t kx = x[int64_t(pos) * xr + P + channel];
      const bf16_t vx = x[int64_t(pos) * xr + 2 * P + channel];
      qv = conv4(qh[0], qh[1], qh[2], qx, weights + channel);
      kv = conv4(kh[0], kh[1], kh[2], kx, weights + 4 * P + channel);
      const float cv =
          conv4(vh[0], vh[1], vh[2], vx, weights + 8 * P + channel);
      v[tid] =
          __bfloat162float(bf16_store(cv * ratio[int64_t(pos) * rr + head]));
      qc[(2 + token) * ct] = qx;
      kc[(2 + token) * ct] = kx;
      vc[(2 + token) * ct] = vx;
      qh[0] = qh[1];
      qh[1] = qh[2];
      qh[2] = qx;
      kh[0] = kh[1];
      kh[1] = kh[2];
      kh[2] = kx;
      vh[0] = vh[1];
      vh[1] = vh[2];
      vh[2] = vx;
      decay[tid] = __expf(
          bound * sigmoid_fast(__expf(a_log[head]) *
                               (bf16_load(g, int64_t(pos) * P + channel) +
                                bias[channel])));
    }
    const auto sums = block_reduce_sum3(qv * qv, kv * kv, 0.f, scratch);
    if (tid < D) {
      q[tid] = qv * rsqrtf(sums.x + 1e-6f) * 0.08838834764831845f;
      k[tid] = kv * rsqrtf(sums.y + 1e-6f);
    }
    __syncthreads();
    const float b = sigmoid_fast(bf16_load(beta, int64_t(pos) * br + head));
    float sumsq = 0.f;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      float parts[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int col = 4 * lane + j * 64 + e;
          h[t][j][e] *= decay[col];
          parts[e] += h[t][j][e] * k[col];
        }
      }
      const float dot =
          row_reduce((parts[0] + parts[1]) + (parts[2] + parts[3]));
      const int row = t * 16 + grp;
      const float update = (v[row] - dot) * b;
      float op[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int col = 4 * lane + j * 64 + e;
          h[t][j][e] += k[col] * update;
          op[e] += h[t][j][e] * q[col];
        }
        store_state(state + int64_t(write) * ss + head * D * D + row * D +
                        4 * lane + j * 64,
                    h[t][j]);
      }
      const float value = row_reduce((op[0] + op[1]) + (op[2] + op[3]));
      if (lane == 0) {
        const float rounded = __bfloat162float(bf16_store(value));
        values[row] = rounded;
        sumsq += rounded * rounded;
      }
    }
    if (lane == 0) scratch[kPartialBase + grp] = sumsq;
    __syncthreads();
    if (tid < 16) {
      const float total = row_reduce(scratch[kPartialBase + tid]);
      if (tid == 0) scratch[kSumsqSlot] = total;
    }
    __syncthreads();
    if (tid < D) {
      const float inverse = rsqrtf(scratch[kSumsqSlot] / D + eps);
      output[int64_t(pos) * P + head * D + tid] = bf16_store(
          values[tid] * inverse * norm[tid] *
          sigmoid_fast(bf16_load(gate, int64_t(pos) * gr + head * D + tid)));
    }
    // Next token reuses both the normalization scratch and shared values.
    __syncthreads();
  }
}
}  // namespace

void forward_sd(torch::Tensor x, torch::Tensor weight, torch::Tensor conv,
                torch::Tensor g, torch::Tensor beta, torch::Tensor ratio,
                torch::Tensor a_log, torch::Tensor bias, torch::Tensor indices,
                torch::Tensor accepted, torch::Tensor cu, torch::Tensor state,
                torch::Tensor gate, torch::Tensor norm, torch::Tensor out,
                double bound, double eps) {
  int tokens = x.size(0), batch = indices.size(0), steps = indices.size(1);
  TORCH_CHECK(steps >= 2 && steps <= 8 && conv.size(2) >= steps + 2,
              "Invalid SD history");
  TORCH_CHECK(indices.scalar_type() == at::kInt && indices.is_contiguous() &&
                  accepted.scalar_type() == at::kInt &&
                  accepted.is_contiguous() && cu.scalar_type() == at::kInt &&
                  cu.is_contiguous(),
              "Invalid SD metadata");
  TORCH_CHECK(accepted.numel() == batch && cu.numel() == batch + 1,
              "Invalid SD metadata shape");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 &&
                  x.size(1) == 3 * P && x.stride(1) == 1,
              "Invalid SD input");
  TORCH_CHECK(g.scalar_type() == at::kBFloat16 && g.is_contiguous() &&
                  g.numel() == tokens * P,
              "Invalid SD decay");
  TORCH_CHECK(beta.scalar_type() == at::kBFloat16 &&
                  beta.sizes() == at::IntArrayRef({1, tokens, H}) &&
                  beta.stride(2) == 1,
              "Invalid SD beta");
  TORCH_CHECK(ratio.scalar_type() == at::kFloat &&
                  ratio.sizes() == beta.sizes() && ratio.stride(2) == 1,
              "Invalid SD ratio");
  TORCH_CHECK(weight.scalar_type() == at::kFloat && weight.is_contiguous() &&
                  weight.sizes() == at::IntArrayRef({3, 4, P}),
              "Invalid SD weight");
  TORCH_CHECK(conv.scalar_type() == at::kBFloat16 && conv.dim() == 3 &&
                  conv.size(1) == 3 * P &&
                  (conv.stride(1) == 1 || conv.stride(2) == 1),
              "Invalid SD conv");
  TORCH_CHECK(state.scalar_type() == at::kFloat,
              "Invalid SD state dtype: expected FP32, got ", state.scalar_type());
  TORCH_CHECK(state.dim() == 4 &&
                  state.sizes() == at::IntArrayRef({conv.size(0), H, D, D}),
              "Invalid SD state shape: expected [", conv.size(0), ", ", H,
              ", ", D, ", ", D, "], got ", state.sizes());
  TORCH_CHECK(state.stride(1) == D * D && state.stride(2) == D &&
                  state.stride(3) == 1 && state.stride(0) >= H * D * D,
              "Invalid SD state stride: expected dense inner dimensions and "
              "outer stride >= ", H * D * D, ", got ", state.strides());
  TORCH_CHECK(state.stride(0) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(state.data_ptr<float>()) % 16 == 0,
              "Invalid SD state alignment: base and outer stride must be "
              "16-byte aligned; outer stride (elements)=", state.stride(0));
  TORCH_CHECK(a_log.scalar_type() == at::kFloat && a_log.is_contiguous() &&
                  a_log.numel() == H && bias.scalar_type() == at::kFloat &&
                  bias.is_contiguous() && bias.numel() == P,
              "Invalid SD parameters");
  TORCH_CHECK(norm.scalar_type() == at::kFloat && norm.is_contiguous() &&
                  norm.numel() == D && gate.scalar_type() == at::kBFloat16 &&
                  gate.sizes() == at::IntArrayRef({tokens, H, D}) &&
                  gate.stride(1) == D && gate.stride(2) == 1,
              "Invalid SD norm");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() &&
                  out.numel() == tokens * P,
              "Invalid SD output");
  for (auto t : {x, weight, conv, g, beta, ratio, a_log, bias, indices,
                 accepted, cu, state, gate, norm, out})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device(),
                "One HIP device required");
  c10::DeviceGuard guard(x.device());
  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  sd_conv_recurrence_norm<<<dim3(H, batch), 256, 0, stream>>>(
      reinterpret_cast<bf16_t*>(x.data_ptr()), weight.data_ptr<float>(),
      reinterpret_cast<bf16_t*>(conv.data_ptr()),
      reinterpret_cast<bf16_t*>(g.data_ptr()),
      reinterpret_cast<bf16_t*>(beta.data_ptr()), ratio.data_ptr<float>(),
      a_log.data_ptr<float>(), bias.data_ptr<float>(), indices.data_ptr<int>(),
      accepted.data_ptr<int>(), cu.data_ptr<int>(), state.data_ptr<float>(),
      reinterpret_cast<bf16_t*>(gate.data_ptr()), norm.data_ptr<float>(),
      reinterpret_cast<bf16_t*>(out.data_ptr()), steps, state.size(0),
      x.stride(0), conv.stride(0), conv.stride(1), conv.stride(2),
      state.stride(0), beta.stride(1), ratio.stride(1), gate.stride(0), bound,
      eps);
  TORCH_CHECK(hipGetLastError() == hipSuccess,
              "Fused native SD HIP launch failed");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward_sd", &forward_sd); }
