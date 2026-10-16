"""CPU-only boundary contracts, without importing the MLX runtime package.

Run: python -m unittest discover -s mlx_vlm/tests -p test_mtp_boundary_contracts.py
Native splitter/generation regression tests remain in test_speculative.py.
"""

import gc
import importlib.util
import sys
import unittest
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def _load_contract(name, relative_path):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


layout_module = _load_contract("_mtp_layout_contract", "speculative/prompt_layout.py")
tensor_module = _load_contract(
    "_glm_tensor_contract", "speculative/drafters/glm5_next_mtp/tensor_provenance.py"
)
PromptLayout = layout_module.PromptLayout
TargetPrefillResult = layout_module.TargetPrefillResult
SplitTensor = tensor_module.SplitTensor


class PromptLayoutTests(unittest.TestCase):
    def test_chunked_prefill_keeps_original_and_residual_padding_separate(self):
        layout = PromptLayout((42, 18), 3, (4, 0), 3)
        self.assertEqual(layout.original_left_padding, (4, 0))
        self.assertEqual(layout.residual_left_padding, (1, 0))
        self.assertEqual(layout.row_ids, (42, 18))

    def test_padding_consumed_in_earlier_columns_is_zero(self):
        self.assertEqual(
            PromptLayout((1, 2), 2, (2, 4), 6).residual_left_padding, (0, 0)
        )

    def test_unknown_padding_is_explicit_and_preserves_unpadded_fallback(self):
        layout = PromptLayout((1, 2), 3, None, None)
        self.assertIsNone(layout.original_left_padding)
        self.assertEqual(layout.residual_left_padding, (0, 0))

    def test_invalid_layouts_rejected(self):
        for args in [
            ((), 2, (), 0),
            ((1, 1), 2, (0, 0), 0),
            ((1,), -1, (0,), 0),
            ((1,), 2, None, 0),
            ((1,), 2, (0,), None),
            ((1, 2), 2, (0,), 0),
            ((1,), 2, (3,), 0),
            ((1,), 2, (-1,), 0),
            ((1,), 2, (0,), -1),
        ]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                PromptLayout(*args)

    def test_target_prefill_keeps_tokens_hidden_coverage_and_reuse_mode(self):
        layout = PromptLayout((5, 8), 3, (1, 0), 0)
        tokens = SimpleNamespace(shape=(2, 3))
        hidden = SimpleNamespace(shape=(2, 3, 16))
        result = TargetPrefillResult(layout, tokens, hidden, "cold_only")
        self.assertIs(result.hidden, hidden)
        self.assertIs(result.prompt_tokens, tokens)
        self.assertEqual(result.hidden_columns, 3)
        self.assertEqual(result.cache_reuse, "cold_only")

    def test_prefill_rejects_wrong_rows_or_missing_or_unaligned_hidden_columns(self):
        layout = PromptLayout((5, 8), 3, (1, 0), 0)
        for token_shape, hidden_shape in [
            ((1, 3), (2, 3, 16)),
            ((2, 4), (2, 3, 16)),
            ((2, 3), (1, 3, 16)),
            ((2, 3), (2, 1, 16)),
            ((2, 3), (2, 4, 16)),
            ((2, 3), (2, 3)),
        ]:
            with self.subTest(token_shape=token_shape, hidden_shape=hidden_shape):
                with self.assertRaises(ValueError):
                    TargetPrefillResult(
                        layout,
                        SimpleNamespace(shape=token_shape),
                        SimpleNamespace(shape=hidden_shape),
                        "cold_only",
                    )


class TensorProvenanceTests(unittest.TestCase):
    def fp8(self, source="kv_b", value=None):
        return SplitTensor(source, value or object(), (source,), "fp8_e4m3", (128, 128))

    def test_projection_split_carries_encoding_scale_layout_and_source(self):
        fused = self.fp8()
        q = fused.transformed("embed_q", object())
        out = fused.transformed("unembed_out", object())
        for result in (q, out):
            self.assertEqual(result.source_keys, ("kv_b",))
            self.assertEqual(result.source_encoding, "fp8_e4m3")
            self.assertEqual(result.scale_layout, (128, 128))
            self.assertIsNot(result.value, fused.value)
        self.assertEqual(q.destination, "embed_q")
        self.assertEqual(out.destination, "unembed_out")

    def test_stacked_experts_keep_each_source(self):
        value = object()
        records = [self.fp8("expert.0"), self.fp8("expert.1")]
        result = SplitTensor.stacked("switch_mlp", value, records)
        self.assertIs(result.value, value)
        self.assertEqual(result.source_keys, ("expert.0", "expert.1"))
        self.assertEqual(result.source_encoding, "fp8_e4m3")

    def test_mixed_expert_encoding_is_rejected_instead_of_losing_fp8_provenance(self):
        sources = [self.fp8(), SplitTensor("dense", object(), ("dense",))]
        with self.assertRaisesRegex(ValueError, "mixed source encodings"):
            SplitTensor.stacked("switch_mlp", object(), sources)

    def test_changed_or_missing_value_cannot_reuse_stale_provenance(self):
        record = self.fp8()
        record.require_current_value({record.destination: record.value})
        for tensors in ({}, {record.destination: object()}):
            with self.assertRaises(ValueError):
                record.require_current_value(tensors)

    def test_packing_changes_current_encoding_without_losing_source(self):
        source = self.fp8()
        packed = source.transformed("packed", object(), value_encoding="mlx_mxfp8")
        self.assertEqual(packed.source_encoding, "fp8_e4m3")
        self.assertEqual(packed.value_encoding, "mlx_mxfp8")
        self.assertEqual(packed.scale_layout, (128, 128))

    def test_rejects_unsupported_source_encoding_and_scale_layout(self):
        for encoding, layout in [
            ("fp8_e4m3", None),
            ("fp8_e4m3", (32, 32)),
            ("native", (128, 128)),
            ("unknown", None),
        ]:
            with self.subTest(encoding=encoding, layout=layout):
                with self.assertRaises(ValueError):
                    SplitTensor("out", object(), ("source",), encoding, layout)


class _ArrayStandIn:
    """Weak-referenceable storage token, not a numerical MLX substitute."""


def _load_splitter_hook():
    # Import the actual splitter module, replacing only its MLX and framework
    # dependencies. This tests Python reference ownership without Apple hardware.
    module_name = "_glm_lifetime.glm5_next_mtp.split"
    path = (
        Path(__file__).resolve().parents[1]
        / "speculative/drafters/glm5_next_mtp/split.py"
    )
    mx = ModuleType("mlx.core")
    mx.array = _ArrayStandIn
    mx.eval = lambda *args: None
    mx.quantize = lambda value, **kwargs: (_ArrayStandIn(), _ArrayStandIn())
    mlx = ModuleType("mlx")
    mlx.core = mx
    framework = ModuleType("_glm_lifetime.mtp_split")
    framework.MTPSplitter = object
    dependencies = {
        "mlx": mlx,
        "mlx.core": mx,
        "_glm_lifetime.mtp_split": framework,
        "_glm_lifetime.glm5_next_mtp.tensor_provenance": tensor_module,
    }
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module.Glm5NextMTPSplitter, mx


class SplitterLifetimeTests(unittest.TestCase):
    source_config = {
        "model_type": "glm5_next",
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
        },
    }

    def setUp(self):
        self.splitter_class, self.mx = _load_splitter_hook()

    def prepare(self, splitter, *, fp8=True):
        weights = {"mtp.eh_proj.weight": _ArrayStandIn()}
        splitter._tensors = {
            key: SplitTensor(key, value, (key,)) for key, value in weights.items()
        }
        if fp8:
            key = "mtp.self_attn.embed_q.weight"
            weights[key] = _ArrayStandIn()
            splitter._tensors[key] = SplitTensor(
                key, weights[key], ("kv_b",), "fp8_e4m3", (128, 128), "decoded"
            )
        return weights

    def test_fp8_handoff_does_not_pin_packed_or_native_arrays(self):
        splitter = self.splitter_class()
        weights = self.prepare(splitter)
        decoded_ref = weakref.ref(weights["mtp.self_attn.embed_q.weight"])
        splitter.quantization_from_source(weights, self.source_config)
        output_refs = [weakref.ref(value) for value in weights.values()]
        gc.collect()
        self.assertIsNone(decoded_ref())
        self.assertTrue(all(ref() is not None for ref in output_refs))
        del weights
        gc.collect()
        # Keep the splitter alive as conversion/upload orchestration does.
        self.assertIsNotNone(splitter)
        self.assertTrue(all(ref() is None for ref in output_refs))

    def test_native_handoff_does_not_pin_unquantized_arrays(self):
        splitter = self.splitter_class()
        weights = self.prepare(splitter, fp8=False)
        output_refs = [weakref.ref(value) for value in weights.values()]
        self.assertIsNone(splitter.quantization_from_source(weights, {}))
        del weights
        gc.collect()
        self.assertIsNotNone(splitter)
        self.assertTrue(all(ref() is None for ref in output_refs))

    def test_failed_handoff_releases_arrays_and_requires_new_source_provenance(self):
        splitter = self.splitter_class()
        weights = self.prepare(splitter)
        output_refs = [weakref.ref(value) for value in weights.values()]
        with self.assertRaisesRegex(ValueError, "only supports"):
            splitter.quantization_from_source(weights, {})
        with self.assertRaisesRegex(ValueError, "sanitize again"):
            splitter.quantization_from_source(weights, self.source_config)
        del weights
        gc.collect()
        self.assertIsNotNone(splitter)
        self.assertTrue(all(ref() is None for ref in output_refs))

    def test_failed_resanitize_cannot_reclassify_old_decoded_output_as_native(self):
        splitter = self.splitter_class()
        old_weights = self.prepare(splitter)
        with self.assertRaisesRegex(ValueError, "only supports"):
            splitter.quantization_from_source(old_weights, {})
        with self.assertRaisesRegex(ValueError, "Missing FP8 weight"):
            splitter.run_sanitize(
                {"model.layers.1.eh_proj.weight_scale_inv": _ArrayStandIn()},
                {"num_hidden_layers": 1},
            )
        with self.assertRaisesRegex(ValueError, "sanitize again"):
            splitter.quantization_from_source(old_weights, self.source_config)

    def test_successful_resanitize_reenables_source_conversion(self):
        splitter = self.splitter_class()
        with self.assertRaisesRegex(ValueError, "only supports"):
            splitter.quantization_from_source(self.prepare(splitter), {})
        weights = splitter.run_sanitize(
            {"model.layers.1.eh_proj.weight": _ArrayStandIn()},
            {"num_hidden_layers": 1},
        )
        self.assertIsNone(splitter.quantization_from_source(weights, {}))

    def test_native_quantized_handoff_releases_arrays(self):
        splitter = self.splitter_class()
        weights = self.prepare(splitter, fp8=False)
        weights["mtp.eh_proj.scales"] = _ArrayStandIn()
        output_refs = [weakref.ref(value) for value in weights.values()]
        config = {"quantization": {"bits": 8, "group_size": 32, "mode": "mxfp8"}}
        self.assertEqual(
            splitter.quantization_from_source(weights, config), config["quantization"]
        )
        del weights
        gc.collect()
        self.assertIsNotNone(splitter)
        self.assertTrue(all(ref() is None for ref in output_refs))


if __name__ == "__main__":
    unittest.main()
