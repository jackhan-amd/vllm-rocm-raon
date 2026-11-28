# Native-stack validation artifacts

These are the verbatim fixtures and analysis programs used for the 2026-11-28
session. The GPU fixtures expect the pinned experiment layout mounted at
`/work`, including its real checkpoint mounts and output directories. They
are not standalone tests that fetch unspecified model weights.

* `rda_project_ratio_fixture.py`: real BF16 checkpoint weights, native-chain
  and independent FP32 Torch comparison, strided beta, wide gates, changed
  graph inputs. `/work/rda-side` contains the two candidate RDA modules.
* `moe_mhc_fixture.py`: native original chain and independent Torch reference,
  stable expert order, shared output, steering and changed graph replay.
* `sd_bounds_fixture.py`: old failing adapter and repaired adapter loaded from
  `/work/sd-bounds`; reproduces the old guard before proving the repair.
* `geometry_speed_extension.py` and `raon_compile_warmup.py`: exact tested
  process-local policy and compile-only startup warmup. No live state update
  is repeated for compilation.
* `analyze_*.py`: CPU analysis of immutable raw responses, source hashes and
  actual device traces. They never launch inference or synthesize scores.
* `image_manifest.py`: image-build source-integrity check and full manifest.

Run the fixtures with the image's uv and Python interpreter, one process per
allocated GPU; `RAON_REPLICA=0..7` partitions their cases. For example, in the
documented experiment container:

```bash
ROCR_VISIBLE_DEVICES=0 RAON_REPLICA=0 /work/uv run --no-project \
  --python /opt/raon/.venv/bin/python python /work/rda-project-ratio-fixture.py
```

The contract, source archives, all executed server argv, GPU identity, raw gold
records and cleanup receipts are retained in the associated evidence packet.
The source recipe and validation scope are documented in
[`NATIVE-STACK-20261128.md`](../../../docker/raon/NATIVE-STACK-20261128.md).
