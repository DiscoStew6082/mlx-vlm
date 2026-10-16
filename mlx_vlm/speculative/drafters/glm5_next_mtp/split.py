import argparse
import re
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx

from ..mtp_split import MTPSplitter
from .tensor_provenance import SplitTensor

FP8_BLOCK_SIZE = 128
MLX_MXFP8_QUANTIZATION = {"group_size": 32, "bits": 8, "mode": "mxfp8"}


def _dequantize_fine_grained_fp8_weight(
    weight: mx.array,
    scale_inv: mx.array,
    *,
    block_size: int = FP8_BLOCK_SIZE,
) -> mx.array:
    if weight.dtype != mx.uint8 or weight.ndim != 2:
        raise ValueError(
            "GLM fine-grained FP8 weights must be 2D E4M3 bytes loaded as "
            f"uint8; got dtype={weight.dtype}, shape={weight.shape}."
        )
    if scale_inv.ndim != 2 or not mx.issubdtype(scale_inv.dtype, mx.floating):
        raise ValueError(
            "GLM fine-grained FP8 scales must be a 2D floating-point block "
            f"grid; got dtype={scale_inv.dtype}, shape={scale_inv.shape}."
        )

    rows, cols = weight.shape
    expected_scale_shape = (
        (rows + block_size - 1) // block_size,
        (cols + block_size - 1) // block_size,
    )
    if scale_inv.shape != expected_scale_shape:
        raise ValueError(
            "GLM fine-grained FP8 scale shape does not match its weight: "
            f"weight={weight.shape}, scales={scale_inv.shape}, "
            f"expected={expected_scale_shape}."
        )

    pad_rows = (-rows) % block_size
    pad_cols = (-cols) % block_size
    decoded = mx.from_fp8(weight, dtype=mx.bfloat16)
    if pad_rows or pad_cols:
        decoded = mx.pad(decoded, ((0, pad_rows), (0, pad_cols)))

    decoded = decoded.reshape(
        (rows + pad_rows) // block_size,
        block_size,
        (cols + pad_cols) // block_size,
        block_size,
    )
    decoded = (decoded * scale_inv[:, None, :, None]).reshape(
        rows + pad_rows, cols + pad_cols
    )
    return decoded[:rows, :cols]


class Glm5NextMTPSplitter(MTPSplitter):
    # GLM-5-Next stores its nextn layer as a normal decoder layer at index
    # ``num_hidden_layers`` (there is no top-level ``mtp.*`` block), so a normal MLX
    # conversion drops it -- extract it here into a standalone glm5_next_mtp drafter.
    output_model_type = "glm5_next_mtp"
    draft_model_cls = None
    tie_word_embeddings_default = False
    depth_field = "num_nextn_predict_layers"
    block_size_extra = 1
    supports_mlx_source = False

    def __init__(self):
        self._tensors: Dict[str, SplitTensor] = {}
        self._needs_sanitize = False

    def _layer_idx(self, text_config: dict) -> int:
        return int(text_config["num_hidden_layers"])

    def select_keys(self, key: str, text_config: dict) -> bool:
        return f"layers.{self._layer_idx(text_config)}." in key

    def run_sanitize(
        self, tensors: Dict[str, mx.array], text_config: dict
    ) -> Dict[str, mx.array]:
        self._tensors = {}
        # Source provenance is usable only after the complete sanitize succeeds.
        # A failed re-sanitize must not re-enable retries on old decoded values.
        self._needs_sanitize = True
        marker = f"layers.{self._layer_idx(text_config)}."
        records = {
            key: SplitTensor(key, value, (key,)) for key, value in tensors.items()
        }
        scale_keys = sorted(key for key in records if key.endswith(".weight_scale_inv"))
        for scale_key in scale_keys:
            weight_key = scale_key[: -len("_scale_inv")]
            if weight_key not in records:
                raise ValueError(f"Missing FP8 weight for scale tensor {scale_key!r}.")
            restored = _dequantize_fine_grained_fp8_weight(
                records.pop(weight_key).value, records.pop(scale_key).value
            )
            mx.eval(restored)
            records[weight_key] = SplitTensor(
                weight_key, restored, (weight_key,), "fp8_e4m3", (128, 128), "decoded"
            )

        out: Dict[str, SplitTensor] = {}
        for key, record in records.items():
            rest = key[key.index(marker) + len(marker) :]
            if rest.startswith("shared_head.norm."):
                rest = "shared_head_norm." + rest[len("shared_head.norm.") :]
            destination = f"mtp.{rest}"
            out[destination] = record.transformed(destination, record.value)

        # Transform the descriptor alongside the value. Both absorbed projections
        # inherit the fused source's encoding and scale layout automatically.
        kb = "mtp.self_attn.kv_b_proj.weight"
        if kb in out:
            record = out.pop(kb)
            nope = int(text_config["qk_nope_head_dim"])
            vhd = int(text_config["v_head_dim"])
            nheads = int(text_config["num_attention_heads"])
            v = record.value.reshape(nheads, nope + vhd, -1)
            for destination, value in (
                (
                    "mtp.self_attn.embed_q.weight",
                    mx.contiguous(v[:, :nope, :].swapaxes(-1, -2)),
                ),
                ("mtp.self_attn.unembed_out.weight", mx.contiguous(v[:, nope:, :])),
            ):
                out[destination] = record.transformed(destination, value)

        self._tensors = out
        self._needs_sanitize = False
        return {key: record.value for key, record in out.items()}

    def postprocess(self, tensors: Dict[str, mx.array], text_config: dict) -> None:
        # Stack per-expert MoE projections into the SwitchGLU layout when the source
        # ships separate experts (already-stacked checkpoints skip this).
        n_experts = int(
            text_config.get("n_routed_experts", text_config.get("num_experts", 0)) or 0
        )
        if not n_experts:
            return
        pattern = re.compile(
            r"(.*\.experts)\.\d+\.(?:gate_proj|up_proj|down_proj)\.weight$"
        )
        prefixes = {m.group(1) for k in tensors if (m := pattern.match(k))}
        for prefix in prefixes:
            base = prefix[: -len(".experts")]
            for proj in ("gate_proj", "up_proj", "down_proj"):
                keys = [f"{prefix}.{e}.{proj}.weight" for e in range(n_experts)]
                if all(k in tensors for k in keys):
                    stacked_key = f"{base}.switch_mlp.{proj}.weight"
                    records = [self._tensors[k] for k in keys]
                    for record in records:
                        record.require_current_value(tensors)
                    # Validate source compatibility before modifying either map.
                    stacked = SplitTensor.stacked(
                        stacked_key,
                        mx.stack([record.value for record in records]),
                        records,
                    )
                    for key in keys:
                        del tensors[key]
                        del self._tensors[key]
                    self._tensors[stacked_key] = stacked
                    tensors[stacked_key] = stacked.value

    def quantization_from_source(self, tensors, source_config):
        # Descriptors own arrays only while transforming them. A splitter may
        # outlive its returned weights (for example during model-card/upload
        # work), so neither successful handoff nor failure may pin those arrays.
        if self._needs_sanitize:
            raise ValueError(
                "Run GLM sanitize again before retrying a failed conversion"
            )
        try:
            return self._quantization_from_source(tensors, source_config)
        except Exception:
            # A retry must rebuild source provenance instead of treating the
            # partially converted values as a native checkpoint.
            self._needs_sanitize = True
            raise
        finally:
            self._tensors.clear()

    def _quantization_from_source(self, tensors, source_config):
        fp8_keys = [
            key
            for key, record in self._tensors.items()
            if record.source_encoding == "fp8_e4m3"
            and record.value_encoding != "mlx_mxfp8"
        ]
        if any(key.endswith(".scales") for key in tensors):
            if fp8_keys:
                raise ValueError(
                    "Cannot mix native MLX and fine-grained FP8 source tensors"
                )
            return source_config.get("quantization") or source_config.get(
                "quantization_config"
            )
        if not fp8_keys:
            return None
        source_quantization = source_config.get("quantization_config") or {}
        supported = (
            source_config.get("model_type") == "glm5_next"
            and source_quantization.get("quant_method") == "fp8"
            and source_quantization.get("fmt", "e4m3") == "e4m3"
            and source_quantization.get("weight_block_size") == [128, 128]
        )
        if not supported:
            raise ValueError(
                "GLM-5-Next MTP splitting only supports fine-grained E4M3 "
                "FP8 weights with 128x128 blocks."
            )

        for key in fp8_keys:
            self._tensors[key].require_current_value(tensors)
        for weight_key in sorted(fp8_keys):
            record = self._tensors[weight_key]
            quantized = mx.quantize(record.value, **MLX_MXFP8_QUANTIZATION)
            if len(quantized) != 2:
                raise ValueError("MLX MXFP8 quantization unexpectedly produced biases.")
            packed, scales = quantized
            mx.eval(packed, scales)
            tensors[weight_key] = packed
            # Release the dequantized value as each destination is packed; the
            # descriptor keeps source provenance without retaining the large buffer.
            self._tensors[weight_key] = record.transformed(
                weight_key, packed, value_encoding="mlx_mxfp8"
            )
            tensors[weight_key[: -len(".weight")] + ".scales"] = scales
        return dict(MLX_MXFP8_QUANTIZATION)


def split_glm5_next_mtp(
    source: str,
    output: str,
    *,
    revision: Optional[str] = None,
    block_size: Optional[int] = None,
    force_download: bool = False,
) -> Path:
    """Write the GLM-5-Next native nextn layer into a standalone drafter folder."""
    return Glm5NextMTPSplitter().split(
        source,
        output,
        revision=revision,
        block_size=block_size,
        force_download=force_download,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split the GLM-5-Next nextn layer into a standalone MLX drafter."
    )
    parser.add_argument("--model", "--source", dest="source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    output = split_glm5_next_mtp(**vars(args))
    print(f"Wrote GLM-5-Next MTP drafter to {output}")


if __name__ == "__main__":
    main()
