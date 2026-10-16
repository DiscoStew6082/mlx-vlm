"""GLM splitter tensor values and their checkpoint encoding travel together."""

from dataclasses import dataclass, replace
from typing import Any, Literal, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SplitTensor:
    destination: str
    value: Any
    source_keys: Tuple[str, ...]
    source_encoding: Literal["native", "fp8_e4m3"] = "native"
    scale_layout: Optional[Tuple[int, int]] = None
    value_encoding: Literal["source", "decoded", "mlx_mxfp8"] = "source"

    def __post_init__(self):
        if not self.source_keys:
            raise ValueError("A split tensor must retain its source keys")
        if self.value_encoding not in ("source", "decoded", "mlx_mxfp8"):
            raise ValueError("Unknown destination tensor encoding")
        if self.source_encoding == "fp8_e4m3":
            if self.scale_layout != (128, 128):
                raise ValueError("GLM FP8 source tensors require 128x128 scale blocks")
        elif self.source_encoding != "native" or self.scale_layout is not None:
            raise ValueError("Invalid source encoding or scale layout")

    def transformed(
        self, destination: str, value: Any, *, value_encoding: Optional[str] = None
    ) -> "SplitTensor":
        return replace(
            self,
            destination=destination,
            value=value,
            value_encoding=value_encoding or self.value_encoding,
        )

    @classmethod
    def stacked(
        cls, destination: str, value: Any, sources: Sequence["SplitTensor"]
    ) -> "SplitTensor":
        if not sources:
            raise ValueError("Cannot stack an empty tensor list")
        encoding = sources[0].source_encoding
        layout = sources[0].scale_layout
        if any(
            s.source_encoding != encoding
            or s.scale_layout != layout
            or s.value_encoding != sources[0].value_encoding
            for s in sources
        ):
            raise ValueError("Cannot stack GLM experts with mixed source encodings")
        return cls(
            destination,
            value,
            tuple(key for source in sources for key in source.source_keys),
            encoding,
            layout,
            sources[0].value_encoding,
        )

    def require_current_value(self, tensors: dict) -> None:
        if self.destination not in tensors:
            raise ValueError(
                f"Source tensor was lost during MTP conversion: {self.destination!r}."
            )
        if tensors[self.destination] is not self.value:
            raise ValueError(
                "Tensor changed without its source provenance during MTP conversion: "
                f"{self.destination!r}."
            )
