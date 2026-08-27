import argparse
import re
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx

from ..mtp_split import MTPSplitter

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
        self._source_fp8_weight_keys: set[str] = set()

    def _layer_idx(self, text_config: dict) -> int:
        return int(text_config["num_hidden_layers"])

    def select_keys(self, key: str, text_config: dict) -> bool:
        return f"layers.{self._layer_idx(text_config)}." in key

    def run_sanitize(
        self, tensors: Dict[str, mx.array], text_config: dict
    ) -> Dict[str, mx.array]:
        marker = f"layers.{self._layer_idx(text_config)}."
        source_fp8_keys = {
            key[: -len("_scale_inv")]
            for key in tensors
            if key.endswith(".weight_scale_inv")
        }
        tensors = dict(tensors)
        for weight_key in sorted(source_fp8_keys):
            scale_key = f"{weight_key}_scale_inv"
            if weight_key not in tensors:
                raise ValueError(f"Missing FP8 weight for scale tensor {scale_key!r}.")
            restored = _dequantize_fine_grained_fp8_weight(
                tensors.pop(weight_key), tensors.pop(scale_key)
            )
            mx.eval(restored)
            tensors[weight_key] = restored

        out: Dict[str, mx.array] = {}
        mapped_fp8_keys: set[str] = set()
        for key, value in tensors.items():
            rest = key[key.index(marker) + len(marker) :]
            if rest.startswith("shared_head.norm."):
                rest = "shared_head_norm." + rest[len("shared_head.norm.") :]
            out[rest] = value
            if key in source_fp8_keys:
                mapped_fp8_keys.add(f"mtp.{rest}")

        # Absorb the fused MLA kv_b_proj into embed_q / unembed_out (as the base model's
        # Glm5NextSparseAttention expects).
        kb = "self_attn.kv_b_proj.weight"
        if kb in out:
            v = out.pop(kb)
            nope = int(text_config["qk_nope_head_dim"])
            vhd = int(text_config["v_head_dim"])
            nheads = int(text_config["num_attention_heads"])
            v = v.reshape(nheads, nope + vhd, -1)
            out["self_attn.embed_q.weight"] = mx.contiguous(
                v[:, :nope, :].swapaxes(-1, -2)
            )
            out["self_attn.unembed_out.weight"] = mx.contiguous(v[:, nope:, :])
            mapped_kb = f"mtp.{kb}"
            if mapped_kb in mapped_fp8_keys:
                mapped_fp8_keys.remove(mapped_kb)
                mapped_fp8_keys.update(
                    {
                        "mtp.self_attn.embed_q.weight",
                        "mtp.self_attn.unembed_out.weight",
                    }
                )

        self._source_fp8_weight_keys = mapped_fp8_keys
        return {f"mtp.{key}": value for key, value in out.items()}

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
                    tensors[stacked_key] = mx.stack([tensors.pop(k) for k in keys])
                    if all(k in self._source_fp8_weight_keys for k in keys):
                        self._source_fp8_weight_keys.difference_update(keys)
                        self._source_fp8_weight_keys.add(stacked_key)

    def quantization_from_source(self, tensors, source_config):
        if any(key.endswith(".scales") for key in tensors):
            return source_config.get("quantization") or source_config.get(
                "quantization_config"
            )
        if not self._source_fp8_weight_keys:
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

        for weight_key in sorted(self._source_fp8_weight_keys):
            if weight_key not in tensors:
                raise ValueError(
                    f"FP8 source weight was lost during MTP conversion: {weight_key!r}."
                )
            quantized = mx.quantize(tensors[weight_key], **MLX_MXFP8_QUANTIZATION)
            if len(quantized) != 2:
                raise ValueError("MLX MXFP8 quantization unexpectedly produced biases.")
            packed, scales = quantized
            mx.eval(packed, scales)
            tensors[weight_key] = packed
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
