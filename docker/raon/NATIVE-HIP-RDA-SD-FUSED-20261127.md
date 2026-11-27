# Single-kernel native HIP RDA SD candidate

The `rocm_hip_rda_sd_fused.hip` candidate combines accepted-history convolution,
RDA recurrence, every FP32 candidate checkpoint and gated output RMSNorm in one
HIP launch per layer. It shares Kimi HIP's DPP/float4 helpers with native AR.
There is no hipify, PTX or Triton compute in the selected operation. A single
CTA owns one sequence/head and retains convolution history and recurrent state
across tokens. No packed/intermediate allocation or separate initialization
kernel is needed: this kernel initializes its NULL suffix output itself.

Separate erase/write gates and BF16 conv/SiLU, post-ratio V and pre-norm output
boundaries remain. Each accepted count selects its own recurrent checkpoint and
three convolution taps. Every candidate checkpoint is still written. The
operation excludes projection and ratio calculation. It does not eliminate the
logical 20 MiB of FP32 checkpoint writes for an eight-token, 40-head request.

## Finite validation and warmed graph timing

Job `89711`, TheraC80, GPU0, MI355X, Torch `2.14.0+rocm10.2.0`. An isolated
source snapshot/private loader selected `load("sd_fused")` for SD and retained
native AR. Initial batch 1/2/4/8/16/32 gold passed 102 cases. After updating the
shared AR helper to the final root version, batch1/32 and the kernel-witness
run passed again. Cases cover AR capture/replay and SD lengths 2/4/8 with all
accepted counts, reordered state indices, complete convolution histories,
untouched recurrent rows, independent Torch output/state gold and exact
graph/direct output/cache checks. Existing tolerance was not relaxed.

| Batch | Verify length | FlyDSL adapter (us) | Fused HIP adapter (us) | Speedup |
| --- | --- | ---: | ---: | ---: |
| 1 | 2 | 18.931 | 8.771 | 2.16x |
| 1 | 4 | 24.963 | 15.247 | 1.64x |
| 1 | 8 | 36.709 | 27.508 | 1.33x |
| 32 | 2 | 88.317 | 52.896 | 1.67x |
| 32 | 4 | 163.419 | 88.738 | 1.84x |
| 32 | 8 | 320.497 | 148.161 | 2.16x |

Each successful condition has one CUDA-event window: 32 operations per graph,
20 separate warmup replays, then 100 measured replays. Compilation and gold
checks precede timing. Only disposable fixture caches are advanced. Kernel
witness collection is separate from timing and names only
`sd_conv_recurrence_norm` for native SD. The profiler has finished; no profiling
overhead is included in these graph timings. These are single-layer adapter
costs, not output tokens/sec, model accuracy or full-model gain. Single-layer
cache reuse is more favorable than full-model state/weight traffic.

The old three-stage HIP SD reference measured about 35.79 us at batch1/length8
and 182.40 us at batch32/length8 in job89710. Those are historical references,
not a paired same-run comparison. The paired table above uses FlyDSL.

## Why batch1 still falls short of 3x

The final gfx950 compiled function reports 190 VGPRs, 87 SGPRs, zero AGPRs,
2,832 bytes LDS, zero private segment, zero VGPR/SGPR spills, 256 threads and
wave64. The binary metadata therefore gives no evidence that spilling is the
principal problem. High register use can still constrain occupancy; no measured
occupancy claim follows from these counts alone.

The actual GPU reports 256 CUs. At batch1 the grid has only 40 CTAs, bounding
simultaneously occupied CUs to at most 40/256 (15.625%). This is a grid-size
upper bound, not a CU-busy counter. Batch1 latency rises by 3.238 us/token from
length2 to4, and 3.065 us/token from4 to8. Fixed launch overhead alone cannot
explain that slope. Token dependencies, synchronization, arithmetic and state
traffic remain; their individual costs have not yet been measured.

The next candidate is a shared conv/QK-normalization/decay prologue followed by
V-split native HIP recurrence and norm. Eight independent 16-V-row CTAs/head
would provide 320 CTAs at batch1 while computing common gates once. It must
retain every checkpoint and avoid duplicate convolution-cache writes. That
candidate is a hypothesis until gold and timing establish its behavior.

## Artifacts and limits

The [receipt](../../tools/raon_validation/results/native-hip-rda-sd-fused-20261127.json)
records source hashes, final binary module `55743662c0b57f2ce5fd`, gold,
witnesses and timing. Raw files are under
`/mnt/hammerspace/jackhan/raon-native-mixed-20261127/sd-fused` and local
`.amd-validation/sd-small-batch-fusion-20261127`. ELF notes and the original
source/helper snapshots are preserved. An initial setup attempt failed platform
preflight because it dropped image PYTHONPATH; that log is also retained.

The production opt-in selector and expanded-pool AR/mixed dispatch have separate
integration acceptance. This standalone receipt does not validate arbitrary
ragged/NULL-suffix SD, mixed-mode shared layers, or real-model MXFP4/MXFP8 quality.
Keep the candidate opt-in and do not silently change active serving profiles.
