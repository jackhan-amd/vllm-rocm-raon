/* SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 * Experimental shared prologue, V-split native recurrence and gated norm.
 */
#define RAON_RDA_NO_MODULE
#include "rocm_hip_rda_ar.hip"

namespace {
constexpr int H = 40, D = 128, P = H * D;

template <int Steps>
__global__ void sd_common_prologue(
    const bf16_t* x, const float* weights, bf16_t* conv, const bf16_t* g,
    const float* ratio, const float* a_log, const float* bias, const int* idx,
    const int* accepted, const int* cu, float* common, bf16_t* intermediate,
    int slots, int64_t xr, int64_t cs, int64_t cc, int64_t ct, int64_t rr,
    float bound) {
  const int n = blockIdx.y, head = blockIdx.x;
  const int token = threadIdx.x / D, channel = threadIdx.x % D;
  const int lo = cu[n], span = cu[n + 1] - lo, a = accepted[n];
  for (int t = token; t < span; t += Steps)
    intermediate[int64_t(lo + t) * P + head * D + channel] = bf16_store(0.f);
  if (a < 1 || a > Steps) return;
  const int first = idx[n * Steps], read = idx[n * Steps + a - 1];
  if (first <= 0 || first >= slots || read <= 0 || read >= slots) return;
  int count = min(span, Steps);
  for (int t = 0; t < count; ++t) {
    const int write = idx[n * Steps + t];
    if (write <= 0 || write >= slots) {
      count = t;
      break;
    }
  }
  if (!count) return;
  const bool live = token < count;
  const int local = head * D + channel;
  auto* qc = conv + int64_t(first) * cs + local * cc;
  auto* kc = qc + P * cc;
  auto* vc = kc + P * cc;
  bf16_t qh[3], kh[3], vh[3];
  bf16_t qx = bf16_store(0.f), kx = qx, vx = qx;
  float qv = 0.f, kv = 0.f, vv = 0.f;
  if (live) {
    const int pos = lo + token;
#pragma unroll
    for (int w = 0; w < 3; ++w) {
      const int previous = token + w;
      if (previous < 3) {
        qh[w] = qc[(a - 1 + previous) * ct];
        kh[w] = kc[(a - 1 + previous) * ct];
        vh[w] = vc[(a - 1 + previous) * ct];
      } else {
        const int64_t row = int64_t(lo + previous - 3) * xr + local;
        qh[w] = x[row];
        kh[w] = x[row + P];
        vh[w] = x[row + 2 * P];
      }
    }
    qx = x[int64_t(pos) * xr + local];
    kx = x[int64_t(pos) * xr + P + local];
    vx = x[int64_t(pos) * xr + 2 * P + local];
#pragma unroll
    for (int w = 0; w < 3; ++w) {
      qv += __bfloat162float(qh[w]) * weights[w * P + local];
      kv += __bfloat162float(kh[w]) * weights[4 * P + w * P + local];
      vv += __bfloat162float(vh[w]) * weights[8 * P + w * P + local];
    }
    qv += __bfloat162float(qx) * weights[3 * P + local];
    kv += __bfloat162float(kx) * weights[7 * P + local];
    vv += __bfloat162float(vx) * weights[11 * P + local];
    qv = __bfloat162float(bf16_store(silu_fast(qv)));
    kv = __bfloat162float(bf16_store(silu_fast(kv)));
    vv = __bfloat162float(bf16_store(silu_fast(vv)));
  }
  __shared__ float qparts[Steps][8], kparts[Steps][8];
  __shared__ float qnorm[Steps], knorm[Steps];
  const int lane = channel & 15, group = channel / 16;
  const float qs = row_reduce(qv * qv), ks = row_reduce(kv * kv);
  if (lane == 0) {
    qparts[token][group] = qs;
    kparts[token][group] = ks;
  }
  // All token groups finish reading old history before any cache write.
  __syncthreads();
  if (channel < 16) {
    const float qtotal = row_reduce(channel < 8 ? qparts[token][channel] : 0.f);
    const float ktotal = row_reduce(channel < 8 ? kparts[token][channel] : 0.f);
    if (channel == 0) {
      qnorm[token] = rsqrtf(qtotal + 1e-6f) * 0.08838834764831845f;
      knorm[token] = rsqrtf(ktotal + 1e-6f);
    }
  }
  __syncthreads();
  if (live) {
    const int pos = lo + token;
    common[int64_t(pos) * 4 * P + local] = qv * qnorm[token];
    common[int64_t(pos) * 4 * P + P + local] = kv * knorm[token];
    common[int64_t(pos) * 4 * P + 2 * P + local] = __expf(
        bound *
        sigmoid_fast(__expf(a_log[head]) *
                     (bf16_load(g, int64_t(pos) * P + local) + bias[local])));
    common[int64_t(pos) * 4 * P + 3 * P + local] =
        __bfloat162float(bf16_store(vv * ratio[int64_t(pos) * rr + head]));
    if (token == 0) {
      qc[0] = qh[1];
      qc[ct] = qh[2];
      kc[0] = kh[1];
      kc[ct] = kh[2];
      vc[0] = vh[1];
      vc[ct] = vh[2];
    }
    qc[(2 + token) * ct] = qx;
    kc[(2 + token) * ct] = kx;
    vc[(2 + token) * ct] = vx;
  }
}

__global__ __launch_bounds__(256) void sd_vsplit_recurrence(
    const float* common, const bf16_t* beta, const int* idx,
    const int* accepted, const int* cu, float* state, bf16_t* output, int steps,
    int slots, int64_t ss, int64_t br) {
  const int head = blockIdx.x, n = blockIdx.y, tile = blockIdx.z,
            tid = threadIdx.x;
  const int a = accepted[n], lo = cu[n], hi = cu[n + 1];
  if (a < 1 || a > steps) return;
  const int first = idx[n * steps], read = idx[n * steps + a - 1];
  if (first <= 0 || first >= slots || read <= 0 || read >= slots) return;
  const int lane = tid & 15, row = tile * 16 + tid / 16;
  f32x4 h[2];
#pragma unroll
  for (int j = 0; j < 2; ++j)
    h[j] = load_state(state + int64_t(read) * ss + head * D * D + row * D +
                      4 * lane + j * 64);
  for (int token = 0; token < min(hi - lo, steps); ++token) {
    const int write = idx[n * steps + token];
    if (write <= 0 || write >= slots) break;
    const int pos = lo + token;
    const float* base = common + int64_t(pos) * 4 * P + head * D;
    float q[8], k[8], parts[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int col = 4 * lane + j * 64 + e;
        q[j * 4 + e] = base[col];
        k[j * 4 + e] = base[P + col];
        h[j][e] *= base[2 * P + col];
        parts[e] += h[j][e] * k[j * 4 + e];
      }
    }
    const float dot = row_reduce((parts[0] + parts[1]) + (parts[2] + parts[3]));
    const float update =
        (base[3 * P + row] - dot) *
        sigmoid_fast(bf16_load(beta, int64_t(pos) * br + head));
    float op[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        h[j][e] += k[j * 4 + e] * update;
        op[e] += h[j][e] * q[j * 4 + e];
      }
      store_state(state + int64_t(write) * ss + head * D * D + row * D +
                      4 * lane + j * 64,
                  h[j]);
    }
    const float value = row_reduce((op[0] + op[1]) + (op[2] + op[3]));
    if (lane == 0)
      output[int64_t(pos) * P + head * D + row] = bf16_store(value);
  }
}

__global__ void sd_vsplit_norm(const bf16_t* input, const bf16_t* gate,
                               const float* weight, bf16_t* output, int64_t gr,
                               float eps) {
  const int pos = blockIdx.y, head = blockIdx.x, tid = threadIdx.x;
  __shared__ float scratch[kPartialBase + kRows];
  const float value =
      tid < D ? bf16_load(input, int64_t(pos) * P + head * D + tid) : 0.f;
  const auto sums = block_reduce_sum3(value * value, 0.f, 0.f, scratch);
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
  TORCH_CHECK(state.scalar_type() == at::kFloat && state.is_contiguous() &&
                  state.sizes() == at::IntArrayRef({conv.size(0), H, D, D}),
              "Invalid SD state");
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
  TORCH_CHECK(steps == 2 || steps == 4 || steps == 8,
              "V-split SD supports 2/4/8 slots");
  auto common = torch::empty({tokens, 4 * P}, x.options().dtype(at::kFloat));
  auto intermediate = torch::empty({tokens, P}, x.options());
#define PROLOGUE(S)                                                          \
  sd_common_prologue<S><<<dim3(H, batch), S * D, 0, stream>>>(               \
      reinterpret_cast<bf16_t*>(x.data_ptr()), weight.data_ptr<float>(),     \
      reinterpret_cast<bf16_t*>(conv.data_ptr()),                            \
      reinterpret_cast<bf16_t*>(g.data_ptr()), ratio.data_ptr<float>(),      \
      a_log.data_ptr<float>(), bias.data_ptr<float>(),                       \
      indices.data_ptr<int>(), accepted.data_ptr<int>(), cu.data_ptr<int>(), \
      common.data_ptr<float>(),                                              \
      reinterpret_cast<bf16_t*>(intermediate.data_ptr()), state.size(0),     \
      x.stride(0), conv.stride(0), conv.stride(1), conv.stride(2),           \
      ratio.stride(1), bound)
  switch (steps) {
    case 2:
      PROLOGUE(2);
      break;
    case 4:
      PROLOGUE(4);
      break;
    case 8:
      PROLOGUE(8);
      break;
  }
#undef PROLOGUE
  sd_vsplit_recurrence<<<dim3(H, batch, 8), 256, 0, stream>>>(
      common.data_ptr<float>(), reinterpret_cast<bf16_t*>(beta.data_ptr()),
      indices.data_ptr<int>(), accepted.data_ptr<int>(), cu.data_ptr<int>(),
      state.data_ptr<float>(),
      reinterpret_cast<bf16_t*>(intermediate.data_ptr()), steps, state.size(0),
      state.stride(0), beta.stride(1));
  sd_vsplit_norm<<<dim3(H, tokens), 256, 0, stream>>>(
      reinterpret_cast<bf16_t*>(intermediate.data_ptr()),
      reinterpret_cast<bf16_t*>(gate.data_ptr()), norm.data_ptr<float>(),
      reinterpret_cast<bf16_t*>(out.data_ptr()), gate.stride(0), eps);
  TORCH_CHECK(hipGetLastError() == hipSuccess,
              "V-split native SD HIP launch failed");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward_sd", &forward_sd); }
