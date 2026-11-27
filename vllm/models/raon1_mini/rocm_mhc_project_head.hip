/* SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 * Experimental token-owned mHC project/statistics/chart/aggregate.
 */
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPStream.h>

namespace {
using bf16 = __hip_bfloat16;
constexpr int C = 2560, K = 4 * C, N = 17;
__device__ float wave_sum(float x) {
  for (int offset = 32; offset; offset /= 2) x += __shfl_xor(x, offset, 64);
  return x;
}
__device__ float sig(float x) { return 1.f / (1.f + __expf(-x)); }
__global__ __launch_bounds__(256) void mhc_project_head_single_cta(
    const bf16* s, const bf16* phi, const float* a, const float* b,
    const float* q, const float* h0, const float* neg_h0, const float* eye,
    const float* gate, bf16* aggregate, float* coeff) {
  int row = blockIdx.x, tid = threadIdx.x, lane = tid % 64, wave = tid / 64;
  const bf16* x = s + int64_t(row) * K;
  __shared__ float scores[N], pre[N], ratios[16], inverse;
  for (int pass = 0; pass < 5; ++pass) {
    int col = pass * 4 + wave;
    if (col < N) {
      float total = 0.f;
      for (int split = 0; split < 16; ++split) {
        float partial = 0.f;
        for (int i = 0; i < 10; ++i) {
          int k = split * 640 + lane + i * 64;
          partial +=
              __bfloat162float(x[k]) * __bfloat162float(phi[col * K + k]);
        }
        total += wave_sum(partial);
      }
      if (lane == 0) scores[col] = __bfloat162float(__float2bfloat16(total));
    }
  }
  if (wave == 0) {
    float total = 0.f;
    for (int split = 0; split < 16; ++split) {
      float partial = 0.f;
      for (int i = 0; i < 10; ++i) {
        float value = __bfloat162float(x[split * 640 + lane + i * 64]);
        partial += value * value;
      }
      total += wave_sum(partial);
    }
    if (lane == 0) inverse = rsqrtf(total / float(K) + 1e-5f);
  }
  __syncthreads();
  if (tid < N) pre[tid] = a[tid] * (scores[tid] * inverse) + b[tid];
  __syncthreads();
  float e = 0.f;
  if (tid < 16) {
    int i = tid / 4, j = tid % 4;
    float t0 =
        (q[i * 3] * pre[8] + q[i * 3 + 1] * pre[11]) + q[i * 3 + 2] * pre[14];
    float t1 =
        (q[i * 3] * pre[9] + q[i * 3 + 1] * pre[12]) + q[i * 3 + 2] * pre[15];
    float t2 =
        (q[i * 3] * pre[10] + q[i * 3 + 1] * pre[13]) + q[i * 3 + 2] * pre[16];
    e = (t0 * q[j * 3] + t1 * q[j * 3 + 1]) + t2 * q[j * 3 + 2];
    ratios[tid] = fmaxf(e / neg_h0[tid], 0.f);
  }
  __syncthreads();
  if (tid < 16) {
    float rho = 0.f;
    for (int i = 0; i < 16; ++i) rho = fmaxf(rho, ratios[i]);
    coeff[row * 20 + 4 + tid] =
        eye[tid] + gate[0] * (h0[tid] + e / (1.f + rho));
  }
  if (tid < 4) coeff[row * 20 + tid] = 2.f * sig(pre[4 + tid]);
  float p0 = sig(pre[0]), p1 = sig(pre[1]), p2 = sig(pre[2]), p3 = sig(pre[3]);
  for (int c = tid; c < C; c += 256) {
    float value = fmaf(p3, __bfloat162float(x[3 * C + c]),
                       fmaf(p2, __bfloat162float(x[2 * C + c]),
                            fmaf(p1, __bfloat162float(x[C + c]),
                                 p0 * __bfloat162float(x[c]))));
    aggregate[int64_t(row) * C + c] = __float2bfloat16(value);
  }
}
}  // namespace

void forward(torch::Tensor s, torch::Tensor phi, torch::Tensor a,
             torch::Tensor b, torch::Tensor q, torch::Tensor h0,
             torch::Tensor neg_h0, torch::Tensor eye, torch::Tensor gate,
             torch::Tensor aggregate, torch::Tensor coeff) {
  TORCH_CHECK(s.dim() == 3 && s.size(0) > 0 && s.size(0) <= 65535 &&
                  s.size(1) == 4 && s.size(2) == C &&
                  s.scalar_type() == at::kBFloat16 && s.is_contiguous(),
              "BF16 contiguous [rows,4,2560] required");
  TORCH_CHECK(phi.scalar_type() == at::kBFloat16 && phi.is_contiguous() &&
                  phi.sizes() == at::IntArrayRef({N, K}),
              "BF16 [17,10240] phi required");
  for (auto t : {s, phi, a, b, q, h0, neg_h0, eye, gate, aggregate, coeff})
    TORCH_CHECK(t.is_cuda() && t.device() == s.device() && t.is_contiguous(),
                "One HIP device and contiguous tensors required");
  for (auto t : {a, b, q, h0, neg_h0, eye, gate, coeff})
    TORCH_CHECK(t.scalar_type() == at::kFloat, "FP32 coefficients required");
  TORCH_CHECK(a.numel() == N && b.numel() == N && q.numel() == 12 &&
                  h0.numel() == 16 && neg_h0.numel() == 16 &&
                  eye.numel() == 16 && gate.numel() == 1,
              "Invalid mHC coefficients");
  TORCH_CHECK(aggregate.scalar_type() == at::kBFloat16 &&
                  aggregate.sizes() == at::IntArrayRef({s.size(0), C}) &&
                  coeff.sizes() == at::IntArrayRef({s.size(0), 20}),
              "Invalid mHC outputs");
  c10::DeviceGuard guard(s.device());
  auto stream = c10::cuda::getCurrentCUDAStream(s.get_device()).stream();
  mhc_project_head_single_cta<<<s.size(0), 256, 0, stream>>>(
      reinterpret_cast<bf16*>(s.data_ptr()),
      reinterpret_cast<bf16*>(phi.data_ptr()), a.data_ptr<float>(),
      b.data_ptr<float>(), q.data_ptr<float>(), h0.data_ptr<float>(),
      neg_h0.data_ptr<float>(), eye.data_ptr<float>(), gate.data_ptr<float>(),
      reinterpret_cast<bf16*>(aggregate.data_ptr()), coeff.data_ptr<float>());
  TORCH_CHECK(hipGetLastError() == hipSuccess,
              "mHC project/head HIP launch failed");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward); }
