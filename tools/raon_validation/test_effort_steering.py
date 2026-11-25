# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU request/cache and mixed-batch math contracts for release steering."""

import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch

SOURCE = (
    Path(__file__).resolve().parents[2] / "vllm/models/raon1_mini/effort_steering.py"
)


def module():
    os.environ.pop("RAON_REASONING_EFFORT_STEERING", None)
    spec = importlib.util.spec_from_file_location("effort_under_test", SOURCE)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    value.configure(
        NS(
            reasoning_effort_steering={
                "alpha": {
                    "none": 0,
                    "low": 0.3,
                    "medium": 0.1,
                    "high": 0.05,
                    "xhigh": 0.02,
                    "max": 0,
                },
                "default": "high",
                "layers": [12, 14, 16, 18, 20],
            }
        ),
        "release-test",
    )
    return value


class Steering(unittest.TestCase):
    def setUp(self):
        self.s = module()

    def request(self, effort=None, extra=None):
        request = NS(vllm_xargs=extra, use_beam_search=False)
        serving = NS(
            _effective_chat_template_kwargs=lambda r: (
                {} if effort is None else {"reasoning_effort": effort}
            )
        )
        self.assertIsNone(self.s.api_request_error(serving, request))
        core = NS(sampling_params=NS(extra_args=request.vllm_xargs), cache_salt=None)
        return self.s._engine_request(core)

    def test_api_default_and_all_efforts(self):
        for effort, expected in [
            (None, 0.05),
            ("none", 0),
            ("low", 0.3),
            ("medium", 0.1),
            ("high", 0.05),
            ("xhigh", 0.02),
            ("max", 0),
        ]:
            self.assertEqual(
                self.s.request_alpha(self.request(effort).sampling_params), expected
            )

    def test_cache_identity_and_client_salt(self):
        self.assertNotEqual(
            self.request("low").cache_salt, self.request("high").cache_salt
        )
        self.assertEqual(
            self.request("max").cache_salt, self.request("none").cache_salt
        )
        self.assertNotEqual(self.s.cache_salt(0.05, "a"), self.s.cache_salt(0.05, "b"))
        with self.assertRaises(ValueError):
            self.s.cache_salt(0, "raon-effort-forged")

    def test_untrusted_alpha_is_dropped(self):
        core = NS(
            sampling_params=NS(
                extra_args={self.s.ALPHA_KEY: 999, self.s._MARK: "forged", "other": 1}
            ),
            cache_salt=None,
        )
        self.s._engine_request(core)
        self.assertEqual(core.sampling_params.extra_args, {"other": 1})

    def test_unknown_effort_and_beam_rejected(self):
        request = NS(vllm_xargs=None, use_beam_search=True)
        serving = NS(_effective_chat_template_kwargs=lambda r: {})
        self.assertIn("beam", self.s.api_request_error(serving, request))
        serving._effective_chat_template_kwargs = lambda r: {"reasoning_effort": "typo"}
        self.assertIn("reasoning_effort", self.s.api_request_error(serving, request))

    def test_mixed_token_mapping_and_stale_padding(self):
        self.s.G = torch.zeros(16)
        runner = NS()
        self.s.fill(
            runner,
            np.array([0.3, 0, 0.05], np.float32),
            torch.tensor([0, 2, 3, 7], dtype=torch.int32),
            12,
        )
        expected = torch.tensor([0.3, 0.3, 0, 0.05, 0.05, 0.05, 0.05] + [0] * 9)
        torch.testing.assert_close(self.s.G, expected, rtol=0, atol=0)
        self.s.fill(
            runner,
            np.array([0.02], np.float32),
            torch.tensor([0, 1], dtype=torch.int32),
            4,
        )
        torch.testing.assert_close(self.s.G[1:], torch.zeros(15), rtol=0, atol=0)
        self.s.fill(runner, np.zeros(1, np.float32), torch.tensor([0, 1]), 4)
        self.assertEqual(self.s.G.count_nonzero().item(), 0)

    def test_gold_and_zero_alpha_preserves_signed_zero(self):
        h = torch.tensor([[0.0, -0.0, 2.0], [1.0, 2.0, 3.0]], dtype=torch.bfloat16)
        c = torch.tensor([2.0, -4.0, 8.0])
        self.s.G = torch.tensor([0.0, 0.05])
        out = self.s.steer(h, c)
        torch.testing.assert_close(
            out[1], torch.tensor([1.1, 1.8, 3.4]).bfloat16(), rtol=0, atol=0
        )
        self.assertTrue(torch.equal(out[0].view(torch.int16), h[0].view(torch.int16)))

    def test_v2_reordered_requests_and_slot_reuse(self):
        self.s.G = torch.zeros(16)
        runner = NS(max_num_reqs=8, req_states=NS(req_id_to_index={"a": 4, "b": 1}))
        self.s.runner_v2_add_requests(
            runner,
            NS(
                scheduled_new_reqs=[
                    NS(req_id="a", sampling_params=self.request("low").sampling_params),
                    NS(
                        req_id="b", sampling_params=self.request("high").sampling_params
                    ),
                ]
            ),
        )
        batch = NS(
            idx_mapping_np=np.array([1, 4]),
            num_reqs=2,
            query_start_loc=torch.tensor([0, 1, 3]),
            num_tokens_after_padding=8,
        )
        self.s.runner_v2_prepare_inputs(runner, batch)
        torch.testing.assert_close(
            self.s.G[:3], torch.tensor([0.05, 0.3, 0.3]), rtol=0, atol=0
        )
        self.s.runner_v2_add_requests(
            runner,
            NS(
                scheduled_new_reqs=[
                    NS(req_id="b", sampling_params=self.request("max").sampling_params)
                ]
            ),
        )
        self.assertEqual(runner._raon_effort_alpha[1], 0)

    def test_vector_loading_and_completeness(self):
        self.s.G = torch.zeros(16)
        self.s._vectors = {12: torch.zeros(3)}
        with self.assertRaises(AssertionError):
            self.s.check_loaded()
        self.s.load_vector(12, torch.tensor([1.0, 2.0, 3.0]))
        ordinary = [
            ("model.embed_tokens.weight", torch.ones(2)),
            ("model.layers.12.moe.reasoning_effort_steering", torch.ones(3)),
        ]
        self.assertEqual(
            [k for k, _ in self.s.take_vectors(ordinary)], [ordinary[0][0]]
        )
        with self.assertRaises(AssertionError):
            self.s.load_vector(12, torch.tensor([float("nan"), 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
