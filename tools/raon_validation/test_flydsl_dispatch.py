# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only adapter/control-flow checks; these do not validate GPU arithmetic."""

import ast
import functools
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / "vllm/models/raon1_mini"
PACKAGE = "_raon_flydsl_test"


def load(name):
    spec = importlib.util.spec_from_file_location(
        PACKAGE + "." + name, SOURCE / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    setattr(sys.modules[PACKAGE], name, module)
    return module


class Tensor:
    def __init__(self, shape, dtype="bf16", value=0):
        self.shape, self.dtype, self.value = shape, dtype, value
        self.device, self.is_cuda = "cuda:0", True
        self.ndim = len(shape)
        self._version = 0

    def to(self, dtype):
        return Tensor(self.shape, dtype, self.value)

    def float(self):
        return self.to("f32")

    def bfloat16(self):
        return self.to("bf16")

    def view(self, *shape):
        size = 1
        for dim in self.shape:
            size *= dim
        known = 1
        for dim in shape:
            if dim != -1:
                known *= dim
        return Tensor(tuple(size // known if d == -1 else d for d in shape), self.dtype)

    reshape = view

    def stride(self):
        result, size = [], 1
        for dim in reversed(self.shape):
            result.append(size)
            size *= dim
        return tuple(reversed(result))

    def data_ptr(self):
        return id(self)

    def clone(self):
        return Tensor(self.shape, self.dtype, self.value)

    def copy_(self, other):
        self.value = other.value
        self._version += 1
        return self


class FlyDSLDispatch(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules)
        self.modules.start()
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(SOURCE)]
        sys.modules[PACKAGE] = package
        self.cuda = types.SimpleNamespace(
            get_device_properties=lambda _: types.SimpleNamespace(gcnArchName="gfx950"),
            is_current_stream_capturing=lambda: False,
        )
        sys.modules["torch"] = types.SimpleNamespace(
            version=types.SimpleNamespace(hip="10.2"),
            cuda=self.cuda,
            bfloat16="bf16",
            int32="i32",
        )
        self.backend = load("mxfp8_flydsl")
        self.calls = []
        self.q13 = Tensor((128, 2560, 2560))
        self.q2 = Tensor((128, 2560, 1280))
        self.backend.packed_weights = lambda *_: (self.q13, "s13", self.q2, "s2")

        def sort(*args, **kwargs):
            self.calls.append(("sort", args, kwargs))
            return "sorted_ids", "sorted_weights", "expert_ids", "valid", None

        def quant(x, **kwargs):
            self.calls.append(("quant", x.shape, kwargs))
            return x, "activation_scales"

        def stage1(*args, **kwargs):
            self.calls.append(("stage1", args, kwargs))
            if kwargs["out_dtype"] == "fp8":
                return Tensor((8, 6, 1280)), "fused_a2_scales"
            return Tensor((8, 6, 1280))

        def stage2(*args, **kwargs):
            self.calls.append(("stage2", args, kwargs))
            return Tensor((8, 6, 2560))

        self.backend._ops = lambda: (sort, quant, stage1, stage2, None, None)

    def tearDown(self):
        self.modules.stop()

    def invoke(self):
        return self.backend.expert_rows(
            Tensor((8, 2560)),
            Tensor((8, 6)),
            Tensor((8, 6), "i32"),
            types.SimpleNamespace(shape=(128, 2176, 2560)),
            types.SimpleNamespace(shape=(128, 2560, 1088)),
            4.0,
            25.0,
        )

    def test_native_pipeline_folds_probability_exactly_once(self):
        result = self.invoke()
        self.assertEqual(result.shape, (8, 6, 2560))
        self.assertEqual(
            [c[0] for c in self.calls], ["sort", "quant", "stage1", "quant", "stage2"]
        )
        s1, s2 = self.calls[2][2], self.calls[4][2]
        self.assertEqual(
            (s1["a_dtype"], s1["b_dtype"], s1["act"]), ("fp8", "fp8", "situv2")
        )
        self.assertEqual((s1["situ_beta"], s1["situ_linear_beta"]), (4.0, 25.0))
        self.assertEqual(s1["sorted_weights"], "sorted_weights")
        self.assertIsNone(s2["sorted_weights"])
        self.assertTrue(s2["return_per_slot"])
        self.assertEqual(self.calls[3][1], (48, 1280))

    def test_fused_a2_uses_stage1_quantization_without_bf16_materialization(self):
        with patch.dict(os.environ, {"RAON_MXFP8_FLYDSL_FUSED_A2": "1"}):
            self.invoke()
        self.assertEqual(
            [c[0] for c in self.calls], ["sort", "quant", "stage1", "stage2"]
        )
        self.assertEqual(self.calls[2][2]["out_dtype"], "fp8")
        self.assertEqual(self.calls[3][2]["a2_scale"], "fused_a2_scales")
        self.assertIsNone(self.calls[3][2]["sorted_weights"])

    def test_other_architecture_fails_before_any_kernel(self):
        self.cuda.get_device_properties = lambda _: types.SimpleNamespace(
            gcnArchName="gfx942"
        )
        with self.assertRaisesRegex(RuntimeError, "gfx950"):
            self.invoke()
        self.assertFalse(self.calls)

    def test_route_shape_mismatch_fails_before_any_kernel(self):
        with self.assertRaisesRegex(ValueError, "Routing"):
            self.backend.expert_rows(
                Tensor((8, 2560)), Tensor((7, 6)), Tensor((8, 6)), None, None, 4, 25
            )
        self.assertFalse(self.calls)

    def test_precision_and_tile_contract(self):
        self.assertEqual(self.backend.padded_intermediate(1088), 1280)
        self.assertEqual(self.backend.padded_intermediate(1280), 1280)
        with patch.dict(os.environ, {"RAON_MOE_BACKEND": "flydsl"}):
            self.assertTrue(self.backend.enabled())
        with patch.dict(os.environ, {"RAON_MOE_BACKEND": "legacy"}):
            self.assertFalse(self.backend.enabled())

    def test_mxfp4_requires_native_flydsl_stages(self):
        native = load("rocm_mxfp4_graph")

        def ck_stage():
            pass

        self.assertFalse(native.stage_is_flydsl(ck_stage))
        flydsl_stage = functools.partial(
            ck_stage, kernelName="flydsl_moe2_layout_afp4_wfp4"
        )
        self.assertTrue(native.stage_is_flydsl(flydsl_stage))

    def test_mxfp4_rejects_wrong_quant_method_before_native_call(self):
        native = load("rocm_mxfp4_graph")
        module = types.SimpleNamespace(
            experts=types.SimpleNamespace(
                routed_experts=types.SimpleNamespace(quant_method=object())
            )
        )
        with self.assertRaisesRegex(RuntimeError, "Quark/AITER"):
            native.verify_native_backend(module, Tensor((8, 2560)))

    def test_prefill_explicit_fused_refuses_unavailable_native_kernel(self):
        selector = load("rocm_prefill")
        native = types.ModuleType("vllm.models.kimi_k3.amd.ops.kda_chunk")
        supported = [False]
        native.can_use_fused_kda_chunk = lambda *args: supported[0]
        sys.modules[native.__name__] = native
        utils = types.ModuleType("vllm.third_party.flash_linear_attention.ops.utils")
        utils.FLA_CHUNK_SIZE = 64
        sys.modules[utils.__name__] = utils
        with patch.dict(os.environ, {"RAON_KDA_PREFILL": "fused"}):
            with self.assertRaisesRegex(RuntimeError, "native HIP"):
                selector.select_backend("auto", 128, "bf16", -5)
            supported[0] = True
            self.assertEqual(
                selector.select_backend("auto", 128, "bf16", -5), "rocm_fused"
            )
        with patch.dict(os.environ, {"RAON_KDA_PREFILL": "triton"}):
            self.assertEqual(
                selector.select_backend("fused", 128, "bf16", -5), "rocm_triton"
            )

    def test_compact_loader_allows_flydsl_without_triton_kernel_flag(self):
        tree = ast.parse((SOURCE / "mxfp8_loader.py").read_text(encoding="utf-8-sig"))
        selected = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("compact_loading_enabled", "validate_compact_loading")
        ]
        namespace = {"os": os}
        exec(
            compile(ast.Module(body=selected, type_ignores=[]), "<loader>", "exec"),
            namespace,
        )
        env = {
            "RAON_MXFP8_COMPACT_LOAD": "1",
            "RAON_MOE_BACKEND": "flydsl",
            "RAON_MOE_EXPERT_MXFP8_REALIZED": "1",
            "RAON_MOE_TRAINER_REALIZED": "1",
            "RAON_MOE_MXFP8_W8A8_TRITON": "0",
            "RAON_MOE_MXFP8_W8A8_GEMM_MODE": "0",
            "RAON_MXFP8_MIN_M": "0",
            "RAON_PREFILL_ALIGNED": "0",
            "RAON_MOE_MXFP8_SHADOW": "0",
        }
        with patch.dict(os.environ, env):
            namespace["validate_compact_loading"]()
            with (
                patch.dict(os.environ, {"RAON_MOE_BACKEND": "legacy"}),
                self.assertRaisesRegex(ValueError, "TRITON"),
            ):
                namespace["validate_compact_loading"]()

    def test_unknown_moe_backend_is_not_silent_fallback(self):
        with (
            patch.dict(os.environ, {"RAON_MOE_BACKEND": "typo"}),
            self.assertRaisesRegex(ValueError, "Unknown"),
        ):
            self.backend.enabled()

    def test_explicit_flydsl_decode_overrides_legacy_boolean(self):
        decoder = load("rocm_flydsl_decode")
        with patch.dict(
            os.environ,
            {
                "RAON_ROCM_KDA_DECODE_BACKEND": "flydsl",
                "RAON_ROCM_KDA_FUSED_DECODE": "0",
            },
        ):
            self.assertEqual(decoder.resolve_backend(), "flydsl")
        with (
            patch.dict(os.environ, {"RAON_ROCM_KDA_DECODE_BACKEND": "typo"}),
            self.assertRaisesRegex(ValueError, "Unknown"),
        ):
            decoder.resolve_backend()

    def test_old_decode_boolean_remains_compatible(self):
        decoder = load("rocm_flydsl_decode")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(decoder.resolve_backend(), "legacy")
            os.environ["RAON_ROCM_KDA_FUSED_DECODE"] = "1"
            self.assertEqual(decoder.resolve_backend(), "aiter_triton")

    def test_serving_refuses_compile_only_before_state_or_kernel_work(self):
        decoder = load("rocm_flydsl_decode")
        with (
            patch.dict(os.environ, {"COMPILE_ONLY": "1"}),
            self.assertRaisesRegex(RuntimeError, "COMPILE_ONLY"),
        ):
            decoder.forward(*([None] * 10))


class OuterGraphLifecycle(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules)
        self.modules.start()
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(SOURCE)]
        sys.modules[PACKAGE] = package
        test = self
        self.capturing = False
        self.active_graph = None

        class Graph:
            def replay(self):
                self.output.value = self.input.value * 3

        class Capture:
            def __init__(self, graph):
                self.graph = graph

            def __enter__(self):
                test.capturing, test.active_graph = True, self.graph

            def __exit__(self, *_):
                test.capturing, test.active_graph = False, None

        cuda = types.SimpleNamespace(
            is_current_stream_capturing=lambda: self.capturing,
            CUDAGraph=Graph,
            graph=Capture,
        )
        sys.modules["torch"] = types.SimpleNamespace(
            cuda=cuda,
            bfloat16="bf16",
            empty_like=lambda t: Tensor(t.shape, t.dtype),
        )
        self.graph = load("rocm_moe_outer_graph")
        self.allowed = True
        self.graph.capture_permitted = lambda: self.allowed
        self.owner = types.SimpleNamespace(
            experts=types.SimpleNamespace(
                routed_experts=types.SimpleNamespace(
                    w13_weight=Tensor((128, 2176, 2560)),
                    w2_weight=Tensor((128, 2560, 1088)),
                ),
                router=types.SimpleNamespace(top_k=6),
            ),
            version=1,
        )
        package.model = types.SimpleNamespace(
            moe_trainer_realized=lambda: True,
            moe_expert_mxfp8_realized=lambda: True,
            _MOE_ROUTE_CAPTURE=False,
            raon_phase=types.SimpleNamespace(PREFILL_ALIGNED=False),
            moe_combine_ep_size=lambda: 0,
        )
        self.graph._stamp = lambda m: ((m,), m.version)
        self.computes = 0

    def tearDown(self):
        self.modules.stop()

    def compute(self, module, x):
        self.computes += 1
        out = Tensor(x.shape, value=x.value * 3)
        if self.active_graph is not None:
            self.active_graph.input, self.active_graph.output = x, out
        return out

    def run_graph(self, tokens=8, value=2):
        return self.graph.try_forward(
            self.owner, Tensor((tokens, 2560), value=value), self.compute
        )

    def test_replay_refreshes_inputs_without_reexecuting_python(self):
        self.assertEqual(self.run_graph(value=2).value, 6)
        self.assertEqual(self.computes, 3)  # two warmups and one capture
        self.allowed = False  # replay is legal inside a breakable eager window
        self.assertEqual(self.run_graph(value=5).value, 15)
        self.assertEqual(self.computes, 3)
        self.assertIsNone(self.run_graph(tokens=16))  # no nested new capture
        self.assertEqual(self.computes, 3)

    def test_weight_replacement_invalidates_old_graph(self):
        self.run_graph()
        old = self.owner._raon_moe_outer_cache
        self.owner.version = 2
        self.run_graph()
        self.assertIsNot(old, self.owner._raon_moe_outer_cache)
        self.assertEqual(self.computes, 6)

    def test_capture_error_is_not_silently_hidden(self):
        def fail(*_):
            raise RuntimeError("native kernel failed")

        with self.assertRaisesRegex(RuntimeError, "native kernel failed"):
            self.graph.try_forward(self.owner, Tensor((8, 2560)), fail)

    def test_native_mxfp4_does_not_require_mxfp8_realization(self):
        package = sys.modules[PACKAGE]
        package.model.moe_trainer_realized = lambda: False
        package.model.moe_expert_mxfp8_realized = lambda: False
        self.owner._raon_graph_precision = "mxfp4"
        self.assertEqual(self.run_graph().value, 6)

    def test_small_and_sd_batches_survive_large_shape_warmup(self):
        """The old eight-entry cache starved C=1/8/32 and rejected SD C=32."""
        shapes = [512, 256, 128, 120, 112, 104, 96, 88, 80, 72, 64, 32, 8, 1]
        with patch.dict(
            os.environ,
            {
                "RAON_MOE_OUTER_GRAPH_SHAPES": "64",
                "RAON_MOE_OUTER_GRAPH_MAX_TOKENS": "512",
            },
        ):
            for tokens in shapes:
                self.assertEqual(self.run_graph(tokens=tokens).value, 6)
            count = self.computes
            self.allowed = False
            for tokens in (1, 8, 32, 256):
                self.assertEqual(self.run_graph(tokens=tokens, value=7).value, 21)
            self.assertEqual(self.computes, count)

    def test_replay_guard_restores_depth_after_exception(self):
        observed = []
        graph_module = self.graph

        class Breakable:
            @classmethod
            def current(cls):
                return None

            def replay(self):
                observed.append(graph_module._TLS.replay_depth)
                raise RuntimeError("replay failed")

        sys.modules["vllm.compilation.breakable_cudagraph"] = types.SimpleNamespace(
            BreakableCUDAGraphCapture=Breakable
        )
        guarded = self.graph._breakable_class()
        with self.assertRaisesRegex(RuntimeError, "replay failed"):
            guarded().replay()
        self.assertEqual(observed, [1])
        self.assertEqual(graph_module._TLS.replay_depth, 0)


class KDAStatefulGraphs(unittest.TestCase):
    def setUp(self):
        OuterGraphLifecycle.setUp(self)
        torch = sys.modules["torch"]
        torch.Tensor = Tensor
        torch.version = types.SimpleNamespace(hip="10.2")
        torch.zeros_like = lambda value: Tensor(value.shape, value.dtype)
        self.kda = load("rocm_kda_graph")
        self.kda._stamp = lambda layer: ((layer,), layer.version)
        self.layer = types.SimpleNamespace(
            version=1,
            kda_decode_backend="rocm_recurrent",
            _raon_decode_graph_local=True,
        )
        self.states = {0: 100, 1: 200}
        self.executed = 0
        self.metadata = types.SimpleNamespace(
            num_prefills=0,
            num_decodes=1,
            num_spec_decodes=0,
            num_actual_tokens=1,
            **{name: None for name in self.kda._FIELDS},
        )
        self.metadata.non_spec_state_indices_tensor = Tensor((1,), "i32", 0)
        self.inputs = (Tensor((1, 2), value=2),) * 5
        self.out = Tensor((1, 2))

        class Graph:
            def replay(self):
                self.run()

        torch.cuda.CUDAGraph = Graph

        def step(metadata, *args):
            inputs, out = args[:-1], args[-1]

            def execute():
                self.executed += 1
                if metadata.num_spec_decodes:
                    index = metadata.spec_state_indices_tensor.value
                    accepted = metadata.num_accepted_tokens.value
                else:
                    index = metadata.non_spec_state_indices_tensor.value
                    accepted = 0
                self.states[index] += inputs[0].value + accepted
                out.value = self.states[index]

            if self.capturing:
                self.active_graph.run = execute
            else:
                execute()

        self.layer._forward_step = step

    def tearDown(self):
        self.modules.stop()

    def decode(self):
        return self.kda.try_forward(self.layer, self.metadata, self.inputs, self.out)

    def test_ar_cold_capture_advances_state_once_and_replay_refreshes_row(self):
        self.assertTrue(self.decode())
        self.assertEqual(
            (self.executed, self.out.value, self.states), (1, 102, {0: 102, 1: 200})
        )
        self.inputs[0].value = 5
        self.metadata.non_spec_state_indices_tensor.value = 1
        self.allowed = False
        self.assertTrue(self.decode())
        self.assertEqual(
            (self.executed, self.out.value, self.states), (2, 205, {0: 102, 1: 205})
        )

    def test_small_batches_replay_after_more_than_eight_warmup_shapes(self):
        shapes = [512, 256, 128, 120, 112, 104, 96, 88, 80, 72, 64, 32, 8, 1]

        def select(tokens):
            self.metadata.num_actual_tokens = self.metadata.num_decodes = tokens
            self.metadata.non_spec_state_indices_tensor = Tensor((tokens,), "i32", 0)
            self.inputs = (Tensor((tokens, 2), value=2),) * 5
            self.out = Tensor((tokens, 2))

        with patch.dict(
            os.environ,
            {
                "RAON_KDA_DECODE_GRAPH_SHAPES": "64",
                "RAON_KDA_DECODE_GRAPH_MAX": "512",
            },
        ):
            for tokens in shapes:
                select(tokens)
                self.assertTrue(self.decode())
            self.assertEqual(self.executed, len(shapes))
            self.allowed = False
            for tokens in (1, 8, 32, 256):
                select(tokens)
                self.assertTrue(self.decode())
            self.assertEqual(self.executed, len(shapes) + 4)

    def test_sd_replay_refreshes_acceptance_and_state_indices(self):
        self.metadata.num_decodes = 0
        self.metadata.num_spec_decodes = 1
        self.metadata.spec_state_indices_tensor = Tensor((1, 8), "i32", 0)
        self.metadata.spec_query_start_loc = Tensor((2,), "i32", 0)
        self.metadata.num_accepted_tokens = Tensor((1,), "i32", 1)
        self.assertTrue(self.decode())
        self.assertEqual((self.executed, self.out.value), (1, 103))
        self.inputs[0].value = 5
        self.metadata.spec_state_indices_tensor.value = 1
        self.metadata.num_accepted_tokens.value = 3
        self.assertTrue(self.decode())
        self.assertEqual(
            (self.executed, self.out.value, self.states), (2, 208, {0: 103, 1: 208})
        )

    def test_new_capture_in_breakable_window_leaves_caches_untouched(self):
        self.allowed = False
        self.assertFalse(self.decode())
        self.assertEqual((self.executed, self.states), (0, {0: 100, 1: 200}))

    def test_sd_warmup_preserves_request_vs_query_token_geometry(self):
        with patch.dict(
            os.environ,
            {
                "RAON_ROCM_GRAPH_WARMUP_BATCHES": "1,8,32",
                "RAON_KDA_DECODE_GRAPH_MAX": "512",
            },
        ):
            self.assertEqual(
                self.kda.decode_warmup_batches(8, 128, 16384),
                ((1, 8), (8, 64), (32, 256)),
            )
            self.assertEqual(self.kda.decode_warmup_batches(1, 128, 16384), ())
            with self.assertRaisesRegex(ValueError, "bounds"):
                self.kda.decode_warmup_batches(8, 128, 128)

    def test_prefill_and_mixed_decode_are_not_specialized(self):
        self.metadata.num_prefills = 1
        self.assertFalse(self.decode())
        self.metadata.num_prefills = 0
        self.metadata.num_spec_decodes = 1
        self.assertFalse(self.decode())
        self.assertEqual(self.executed, 0)

    def test_cache_content_changes_do_not_invalidate_weight_stamp(self):
        original_stamp = load("rocm_kda_graph")._stamp
        state = Tensor((2, 4))
        parameter = Tensor((4,))
        layer = types.SimpleNamespace(
            parameters=lambda: [parameter],
            kv_cache=[state],
            gate_lower_bound=-5,
            kda_decode_backend="rocm_recurrent",
            o_norm=types.SimpleNamespace(eps=1e-6),
        )
        _, before = original_stamp(layer)
        state._version += 1
        state.value = 999
        _, after = original_stamp(layer)
        self.assertEqual(before, after)
        parameter._version += 1
        self.assertNotEqual(before, original_stamp(layer)[1])


class NativeSharedStream(unittest.TestCase):
    def setUp(self):
        FlyDSLDispatch.setUp(self)
        self.streams = load("rocm_shared_stream")
        self.cuda.Stream = lambda **kwargs: types.SimpleNamespace(**kwargs)
        self.cuda.Event = object
        vllm = types.ModuleType("vllm")
        vllm.envs = types.SimpleNamespace(VLLM_DISABLE_SHARED_EXPERTS_STREAM=False)
        sys.modules["vllm"] = vllm

        class Shared:
            _output_idx = 0

            def __init__(self):
                self._stream = object()
                self.queued, self.waited, self.allow_queue = 0, 0, True

            def maybe_forward_async(self, inputs):
                if self.allow_queue:
                    self.queued += 1
                return self.allow_queue

            def wait(self):
                self.waited += 1

            def _determine_shared_experts_order(self, inputs):
                return types.SimpleNamespace(name="NO_OVERLAP")

        self.shared = Shared()
        self.module = types.SimpleNamespace(
            _raon_local_graph_world=True,
            experts=types.SimpleNamespace(shared_experts=self.shared),
        )

    def tearDown(self):
        self.modules.stop()

    def test_native_stream_switch_uses_one_native_enqueue_and_wait(self):
        with patch.dict(
            os.environ,
            {"RAON_MOE_BACKEND": "flydsl", "RAON_MOE_SHARED_SIDE_STREAM": "1"},
        ):
            self.streams.configure_native(self.module, Tensor((8, 2560)))
            self.assertTrue(self.shared.maybe_forward_async(Tensor((8, 2560))))
            self.shared.wait()
            self.assertEqual((self.shared.queued, self.shared.waited), (1, 1))
            self.assertEqual(len(self.shared._input_ready_event), 2)
            with (
                patch.dict(os.environ, {"RAON_MOE_SHARED_SIDE_STREAM": "0"}),
                self.assertRaisesRegex(RuntimeError, "Restart"),
            ):
                self.streams.configure_native(self.module, Tensor((8, 2560)))

    def test_native_zero_disables_stream_only_for_candidate(self):
        original = self.shared._stream
        with patch.dict(
            os.environ,
            {"RAON_MOE_BACKEND": "legacy", "RAON_MOE_SHARED_SIDE_STREAM": "0"},
        ):
            self.streams.configure_native(self.module, Tensor((8, 2560)))
            self.assertIs(self.shared._stream, original)
        with patch.dict(
            os.environ,
            {"RAON_MOE_BACKEND": "flydsl", "RAON_MOE_SHARED_SIDE_STREAM": "0"},
        ):
            self.streams.configure_native(self.module, Tensor((8, 2560)))
            self.assertIsNone(self.shared._stream)

    def test_explicit_native_disable_conflict_is_reported(self):
        sys.modules["vllm"].envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM = True
        with (
            patch.dict(
                os.environ,
                {"RAON_MOE_BACKEND": "flydsl", "RAON_MOE_SHARED_SIDE_STREAM": "1"},
            ),
            self.assertRaisesRegex(RuntimeError, "conflicts"),
        ):
            self.streams.configure_native(self.module, Tensor((8, 2560)))

    def test_custom_rocm_stream_switch_applies_inside_capture(self):
        tree = ast.parse((SOURCE / "model.py").read_text(encoding="utf-8-sig"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_raon_fp32_combine_moe_impl_eager"
        )
        condition = next(
            node.test
            for node in function.body
            if isinstance(node, ast.If)
            and "RAON_MOE_SHARED_SIDE_STREAM" in ast.unparse(node.test)
        )
        torch = sys.modules["torch"]
        self.cuda.is_current_stream_capturing = lambda: True
        variables = dict(
            self=types.SimpleNamespace(shared_experts=object()),
            x=Tensor((8, 2560)),
            torch=torch,
            os=os,
        )
        with patch.dict(
            os.environ,
            {
                "RAON_MOE_BACKEND": "flydsl",
                "RAON_MOE_SHARED_SIDE_STREAM": "1",
                "RAON_EXPERIMENT_SHARED_GRAPH": "0",
            },
        ):
            self.assertTrue(
                eval(
                    compile(ast.Expression(condition), "<stream-gate>", "eval"),
                    variables,
                )
            )
            torch.version.hip = None
            self.assertFalse(
                eval(
                    compile(ast.Expression(condition), "<stream-gate>", "eval"),
                    variables,
                )
            )

    def test_custom_zero_keeps_side_stream_off_inside_capture(self):
        tree = ast.parse((SOURCE / "model.py").read_text(encoding="utf-8-sig"))
        function = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "_raon_fp32_combine_moe_impl_eager"
        )
        condition = next(
            n.test
            for n in function.body
            if isinstance(n, ast.If)
            and "RAON_MOE_SHARED_SIDE_STREAM" in ast.unparse(n.test)
        )
        self.cuda.is_current_stream_capturing = lambda: True
        with patch.dict(
            os.environ,
            {"RAON_MOE_BACKEND": "flydsl", "RAON_MOE_SHARED_SIDE_STREAM": "0"},
        ):
            self.assertFalse(
                eval(
                    compile(ast.Expression(condition), "<stream-gate>", "eval"),
                    dict(
                        self=types.SimpleNamespace(shared_experts=object()),
                        x=Tensor((8, 2560)),
                        torch=sys.modules["torch"],
                        os=os,
                    ),
                )
            )


if __name__ == "__main__":
    unittest.main()
