# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export audits must preserve scalar tensors and reject corrupted low-bit data."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from tools.raon_validation.quantized import check


class TestQuantizedExport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source"
        self.model = Path(self.temp.name) / "model"
        self.source.mkdir()
        self.model.mkdir()
        self.weight = "model.layers.1.moe.experts.0.gate_proj.weight"
        self.scalar = "model.layers.1.mhc_alpha"
        source = {
            self.weight: torch.zeros(2, 32, dtype=torch.float8_e4m3fn),
            self.weight + "_scale_inv": torch.full((2, 1), 127, dtype=torch.uint8),
            self.scalar: torch.tensor(0.5, dtype=torch.bfloat16),
        }
        self.output = {
            self.weight: torch.zeros(2, 16, dtype=torch.uint8),
            self.weight + "_scale": torch.full((2, 1), 127, dtype=torch.uint8),
            self.scalar: source[self.scalar].clone(),
        }
        for folder, tensors in ((self.source, source), (self.model, self.output)):
            save_file(tensors, folder / "model.safetensors")
            (folder / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {n: "model.safetensors" for n in tensors}})
            )

    def test_scalar_bf16_excluded_tensor_is_compared_by_its_bytes(self):
        self.assertEqual(check(self.source, self.model)["unchanged_tensors"], 1)
        self.output[self.scalar] += 1
        save_file(self.output, self.model / "model.safetensors")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            check(self.source, self.model)

    def test_nonfinite_e8m0_scale_is_rejected(self):
        self.output[self.weight + "_scale"][0, 0] = 255
        save_file(self.output, self.model / "model.safetensors")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            check(self.source, self.model)

    def test_wrong_packed_width_is_rejected(self):
        self.output[self.weight] = torch.zeros(2, 32, dtype=torch.uint8)
        save_file(self.output, self.model / "model.safetensors")
        with self.assertRaisesRegex(ValueError, "Invalid packed"):
            check(self.source, self.model)


if __name__ == "__main__":
    unittest.main()
