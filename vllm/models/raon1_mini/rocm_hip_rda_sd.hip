/* SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 * Experimental native SD: retain every FP32 checkpoint and accepted history.
 */
#define RAON_RDA_NO_MODULE
#include "rocm_hip_rda_ar.hip"

namespace {
constexpr int H = 40, D = 128, P = H * D;

__global__ void sd_conv(const bf16_t* x, const float* w, bf16_t* cache,
                        const int* idx, const int* accepted, const int* cu,
                        bf16_t* packed, int steps, int slots, int64_t xr,
                        int64_t cs, int64_t cc, int64_t ct) {
  int n = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= 3 * P) return;
  int a = accepted[n], first = idx[n * steps], lo = cu[n], hi = cu[n + 1];
  if (a < 1 || a > steps || first <= 0 || first >= slots) return;
  const int checkpoint = idx[n * steps + a - 1];
  if (checkpoint <= 0 || checkpoint >= slots) return;
  int count = min(hi - lo, steps);
  for (int t = 0; t < count; ++t)
    if (idx[n * steps + t] <= 0 || idx[n * steps + t] >= slots) {
      count = t;
      break;
    }
  if (!count) return;
  auto* cp = cache + int64_t(first) * cs + int64_t(c) * cc;
  // Read all three selected taps before writing the expanded history.
  bf16_t h0 = cp[(a - 1) * ct], h1 = cp[a * ct], h2 = cp[(a + 1) * ct];
  cp[0] = h1;
  cp[ct] = h2;
  int group = c / P, channel = c % P;
  auto* wp = w + group * 4 * P + channel;
  for (int t = 0; t < count; ++t) {
    bf16_t value = x[int64_t(lo + t) * xr + c];
    float acc = 0.f;
    acc += __bfloat162float(h0) * wp[0];
    acc += __bfloat162float(h1) * wp[P];
    acc += __bfloat162float(h2) * wp[2 * P];
    acc += __bfloat162float(value) * wp[3 * P];
    packed[int64_t(lo + t) * 3 * P + c] = bf16_store(silu_fast(acc));
    cp[(2 + t) * ct] = value;
    h0 = h1;
    h1 = h2;
    h2 = value;
  }
}

__global__ __launch_bounds__(256) void sd_recurrence(
    const bf16_t* packed, const bf16_t* g, const bf16_t* beta,
    const float* ratio, const float* a_log, const float* bias, const int* idx,
    const int* accepted, const int* cu, float* state, bf16_t* output, int steps,
    int slots, int64_t ss, int64_t br, int64_t rr, float bound) {
  int n = blockIdx.y, head = blockIdx.x, tid = threadIdx.x;
  int a = accepted[n], lo = cu[n], hi = cu[n + 1];
  if (a < 1 || a > steps) return;
  int first = idx[n * steps], read = idx[n * steps + a - 1];
  if (first <= 0 || first >= slots || read <= 0 || read >= slots) return;
  int lane = tid & 15, grp = tid / 16;
  f32x4 h[8][2];
  for (int t = 0; t < 8; ++t)
    for (int j = 0; j < 2; ++j)
      h[t][j] = load_state(state + int64_t(read) * ss + head * D * D +
                           (t * 16 + grp) * D + 4 * lane + j * 64);
  __shared__ float q[D], k[D], decay[D], v[D], scratch[kPartialBase + kRows];
  int count = min(hi - lo, steps);
  for (int token = 0; token < count; ++token) {
    int write = idx[n * steps + token];
    if (write <= 0 || write >= slots) break;
    int pos = lo + token;
    float qv = 0.f, kv = 0.f;
    if (tid < D) {
      qv = bf16_load(packed, int64_t(pos) * 3 * P + head * D + tid);
      kv = bf16_load(packed, int64_t(pos) * 3 * P + P + head * D + tid);
      float cv =
          bf16_load(packed, int64_t(pos) * 3 * P + 2 * P + head * D + tid);
      v[tid] =
          __bfloat162float(bf16_store(cv * ratio[int64_t(pos) * rr + head]));
      decay[tid] =
          __expf(bound *
                 sigmoid_fast(__expf(a_log[head]) *
                              (bf16_load(g, int64_t(pos) * P + head * D + tid) +
                               bias[head * D + tid])));
    }
    auto sums = block_reduce_sum3(qv * qv, kv * kv, 0.f, scratch);
    if (tid < D) {
      q[tid] = qv * rsqrtf(sums.x + 1e-6f) * 0.08838834764831845f;
      k[tid] = kv * rsqrtf(sums.y + 1e-6f);
    }
    __syncthreads();
    float b = sigmoid_fast(bf16_load(beta, int64_t(pos) * br + head));
    for (int t = 0; t < 8; ++t) {
      float parts[4] = {0.f, 0.f, 0.f, 0.f};
      for (int j = 0; j < 2; ++j)
        for (int e = 0; e < 4; ++e) {
          int col = 4 * lane + j * 64 + e;
          h[t][j][e] *= decay[col];
          parts[e] += h[t][j][e] * k[col];
        }
      float dot = row_reduce((parts[0] + parts[1]) + (parts[2] + parts[3]));
      int row = t * 16 + grp;
      float update = (v[row] - dot) * b;
      float op[4] = {0.f, 0.f, 0.f, 0.f};
      for (int j = 0; j < 2; ++j) {
        for (int e = 0; e < 4; ++e) {
          int col = 4 * lane + j * 64 + e;
          h[t][j][e] += k[col] * update;
          op[e] += h[t][j][e] * q[col];
        }
        store_state(state + int64_t(write) * ss + head * D * D + row * D +
                        4 * lane + j * 64,
                    h[t][j]);
      }
      float value = row_reduce((op[0] + op[1]) + (op[2] + op[3]));
      if (lane == 0)
        output[int64_t(pos) * P + head * D + row] = bf16_store(value);
    }
    __syncthreads();
  }
}

__global__ void sd_norm(const bf16_t* input, const bf16_t* gate,
                        const float* weight, bf16_t* output, int64_t gr,
                        float eps) {
  int pos = blockIdx.y, head = blockIdx.x, tid = threadIdx.x;
  __shared__ float scratch[kPartialBase + kRows];
  float value =
      tid < D ? bf16_load(input, int64_t(pos) * P + head * D + tid) : 0.f;
  auto sums = block_reduce_sum3(value * value, 0.f, 0.f, scratch);
  if (tid < D)
    output[int64_t(pos) * P + head * D + tid] = bf16_store(
        value * rsqrtf(sums.x / D + eps) * weight[tid] *
        sigmoid_fast(bf16_load(gate, int64_t(pos) * gr + head * D + tid)));
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
  for (auto t : {x, weight, conv, g, beta, ratio, a_log, bias, indices, accepted,
                 cu, state, gate, norm, out})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device(),
                "One HIP device required");
  c10::DeviceGuard guard(x.device());
  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  auto packed = torch::zeros({tokens, 3 * P}, x.options());
  auto intermediate = torch::zeros({tokens, P}, x.options());
  sd_conv<<<dim3((3 * P + 255) / 256, batch), 256, 0, stream>>>(
      reinterpret_cast<bf16_t*>(x.data_ptr()), weight.data_ptr<float>(),
      reinterpret_cast<bf16_t*>(conv.data_ptr()), indices.data_ptr<int>(),
      accepted.data_ptr<int>(), cu.data_ptr<int>(),
      reinterpret_cast<bf16_t*>(packed.data_ptr()), steps, state.size(0),
      x.stride(0), conv.stride(0), conv.stride(1), conv.stride(2));
  sd_recurrence<<<dim3(H, batch), 256, 0, stream>>>(
      reinterpret_cast<bf16_t*>(packed.data_ptr()),
      reinterpret_cast<bf16_t*>(g.data_ptr()),
      reinterpret_cast<bf16_t*>(beta.data_ptr()), ratio.data_ptr<float>(),
      a_log.data_ptr<float>(), bias.data_ptr<float>(), indices.data_ptr<int>(),
      accepted.data_ptr<int>(), cu.data_ptr<int>(), state.data_ptr<float>(),
      reinterpret_cast<bf16_t*>(intermediate.data_ptr()), steps, state.size(0),
      state.stride(0), beta.stride(1), ratio.stride(1), bound);
  sd_norm<<<dim3(H, tokens), 256, 0, stream>>>(
      reinterpret_cast<bf16_t*>(intermediate.data_ptr()),
      reinterpret_cast<bf16_t*>(gate.data_ptr()), norm.data_ptr<float>(),
      reinterpret_cast<bf16_t*>(out.data_ptr()), gate.stride(0), eps);
  TORCH_CHECK(hipGetLastError() == hipSuccess, "Native SD HIP launch failed");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward_sd", &forward_sd); }
