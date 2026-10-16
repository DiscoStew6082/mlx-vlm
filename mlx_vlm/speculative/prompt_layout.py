"""Explicit target-prefill layout at the batched MTP drafter handoff.

No MLX import is needed to validate row identities, padding and hidden coverage.
Cache-reuse policy describes the drafter's supported mode; admission remains in
existing generation/server paths, including GLM's full-cold APC fallback.
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple


@dataclass(frozen=True)
class PromptLayout:
    row_ids: Tuple[int, ...]
    token_columns: int
    original_left_padding: Optional[Tuple[int, ...]]
    processed_columns: Optional[int]

    def __post_init__(self):
        if not self.row_ids or len(set(self.row_ids)) != len(self.row_ids):
            raise ValueError("Prompt rows must have distinct identities")
        if self.token_columns < 0:
            raise ValueError("Prompt width cannot be negative")
        if (self.original_left_padding is None) != (self.processed_columns is None):
            raise ValueError(
                "Original padding and processed columns must travel together"
            )
        if self.original_left_padding is not None:
            if len(self.original_left_padding) != len(self.row_ids):
                raise ValueError("Prompt padding must have one value per row")
            if self.processed_columns < 0 or any(
                pad < 0 for pad in self.original_left_padding
            ):
                raise ValueError(
                    "Prompt padding and processed columns cannot be negative"
                )
            if any(pad > self.token_columns for pad in self.residual_left_padding):
                raise ValueError("Residual padding exceeds the prompt suffix width")

    @property
    def residual_left_padding(self) -> Tuple[int, ...]:
        if self.original_left_padding is None:
            # Preserve the existing unpadded fallback, but retain that the cache
            # did not provide layout metadata instead of claiming it did.
            return (0,) * len(self.row_ids)
        return tuple(
            max(pad - self.processed_columns, 0) for pad in self.original_left_padding
        )


@dataclass(frozen=True)
class TargetPrefillResult:
    layout: PromptLayout
    prompt_tokens: Any
    hidden: Any
    cache_reuse: Literal["cold_only", "target_prefix"]

    def __post_init__(self):
        if self.cache_reuse not in ("cold_only", "target_prefix"):
            raise ValueError("Unknown MTP cache-reuse policy")
        token_shape = tuple(self.prompt_tokens.shape)
        hidden_shape = tuple(self.hidden.shape)
        expected = (len(self.layout.row_ids), self.layout.token_columns)
        if token_shape != expected:
            raise ValueError("Prompt tokens do not match their declared row layout")
        if len(hidden_shape) < 3 or hidden_shape[:2] != expected:
            raise ValueError(
                "Target hidden states must cover exactly the prompt suffix"
            )

    @property
    def hidden_columns(self) -> int:
        return int(self.hidden.shape[1])
