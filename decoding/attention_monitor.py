from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class AttentionInfo:
    vector: Optional[torch.Tensor] = None
    focus_pos: Optional[int] = None
    focus_score: float = 0.0
    topk_positions: List[int] = field(default_factory=list)
    topk_scores: List[float] = field(default_factory=list)
    source_len: int = 0
    valid: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    def span_score(self, span: Optional[Tuple[int, int]]) -> float:
        if not self.valid or self.vector is None or span is None:
            return 0.0

        try:
            start, end = span
        except Exception:
            return 0.0

        start = max(0, int(start))
        end = min(int(end), int(self.vector.numel()))

        if end <= start:
            return 0.0

        return float(self.vector[start:end].sum().item())

    def focus_in_span(self, span: Optional[Tuple[int, int]]) -> bool:
        if not self.valid or self.focus_pos is None or span is None:
            return False

        try:
            start, end = span
        except Exception:
            return False

        return int(start) <= int(self.focus_pos) < int(end)

    def topk_intersects_span(self, span: Optional[Tuple[int, int]]) -> bool:
        if not self.valid or span is None:
            return False

        try:
            start, end = span
        except Exception:
            return False

        start = int(start)
        end = int(end)

        for position in self.topk_positions:
            if start <= int(position) < end:
                return True

        return False

    def to_dict(self, compact: bool = True) -> Dict[str, Any]:
        data = {
            "valid": bool(self.valid),
            "focus_pos": self.focus_pos,
            "focus_score": float(self.focus_score),
            "topk_positions": list(self.topk_positions),
            "topk_scores": [float(x) for x in self.topk_scores],
            "source_len": int(self.source_len),
            "meta": dict(self.meta),
        }

        if not compact and self.vector is not None:
            data["vector"] = self.vector.detach().cpu().tolist()

        return data


class AttentionMonitor:
    """
    Lấy cross-attention của decoder tại bước sinh token hiện tại.

    Input thường từ HuggingFace:
        outputs.cross_attentions

    Shape phổ biến:
        [batch, heads, target_len, source_len]

    Output:
        AttentionInfo gồm focus_pos, focus_score, top-k source positions.
    """

    def __init__(
        self,
        top_k: int = 5,
        layer_strategy: str = "last",
        head_strategy: str = "mean",
        normalize: bool = True,
    ):
        self.top_k = max(1, int(top_k))
        self.layer_strategy = str(layer_strategy)
        self.head_strategy = str(head_strategy)
        self.normalize = bool(normalize)
        self.last_debug: Dict[str, Any] = {}

    def get_focus(
        self,
        cross_attentions,
        batch_index: int = 0,
        target_index: int = -1,
        source_attention_mask: Optional[torch.Tensor] = None,
    ) -> AttentionInfo:
        self.last_debug = {
            "available": False,
            "reason": None,
            "layer_strategy": self.layer_strategy,
            "head_strategy": self.head_strategy,
        }

        vector = self.extract_attention_vector(
            cross_attentions=cross_attentions,
            batch_index=batch_index,
            target_index=target_index,
        )

        if vector is None:
            self.last_debug["reason"] = "no_cross_attention"
            return AttentionInfo(valid=False, meta=dict(self.last_debug))

        vector = vector.float()
        vector = torch.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
        vector = torch.clamp(vector, min=0.0)

        if source_attention_mask is not None:
            vector = self._apply_source_mask(
                vector=vector,
                source_attention_mask=source_attention_mask,
                batch_index=batch_index,
            )

            if vector is None:
                self.last_debug["reason"] = "zero_after_source_mask"
                return AttentionInfo(valid=False, meta=dict(self.last_debug))

        total = float(vector.sum().item())

        if total <= 0.0:
            self.last_debug["reason"] = "zero_attention"
            return AttentionInfo(valid=False, meta=dict(self.last_debug))

        if self.normalize:
            vector = vector / total

        source_len = int(vector.numel())

        focus_score_tensor, focus_pos_tensor = torch.max(vector, dim=0)

        focus_pos = int(focus_pos_tensor.item())
        focus_score = float(focus_score_tensor.item())

        k = min(self.top_k, source_len)

        topk_scores_tensor, topk_positions_tensor = torch.topk(
            vector,
            k=k,
            dim=0,
        )

        topk_positions = [
            int(x) for x in topk_positions_tensor.detach().cpu().tolist()
        ]

        topk_scores = [
            float(x) for x in topk_scores_tensor.detach().cpu().tolist()
        ]

        self.last_debug = {
            "available": True,
            "reason": "ok",
            "source_len": source_len,
            "focus_pos": focus_pos,
            "focus_score": focus_score,
            "topk_positions": topk_positions,
            "topk_scores": topk_scores,
        }

        return AttentionInfo(
            vector=vector,
            focus_pos=focus_pos,
            focus_score=focus_score,
            topk_positions=topk_positions,
            topk_scores=topk_scores,
            source_len=source_len,
            valid=True,
            meta=dict(self.last_debug),
        )

    def extract_attention_vector(
        self,
        cross_attentions,
        batch_index: int = 0,
        target_index: int = -1,
    ) -> Optional[torch.Tensor]:
        layers = self._flatten_cross_attentions(cross_attentions)

        if not layers:
            return None

        selected_layers = self._select_layers(layers)

        vectors = []

        for layer in selected_layers:
            vector = self._layer_to_vector(
                layer_attention=layer,
                batch_index=batch_index,
                target_index=target_index,
            )

            if vector is not None:
                vectors.append(vector)

        if not vectors:
            return None

        if len(vectors) == 1:
            return vectors[0]

        return torch.stack(vectors, dim=0).mean(dim=0)

    def _flatten_cross_attentions(self, cross_attentions) -> List[torch.Tensor]:
        if cross_attentions is None:
            return []

        if isinstance(cross_attentions, torch.Tensor):
            return [cross_attentions]

        if isinstance(cross_attentions, (list, tuple)):
            result = []

            for item in cross_attentions:
                result.extend(self._flatten_cross_attentions(item))

            return result

        return []

    def _select_layers(self, layers: List[torch.Tensor]) -> List[torch.Tensor]:
        if not layers:
            return []

        strategy = self.layer_strategy.lower()

        if strategy == "first":
            return [layers[0]]

        if strategy in {"all", "all_mean", "mean"}:
            return layers

        return [layers[-1]]

    def _layer_to_vector(
        self,
        layer_attention: torch.Tensor,
        batch_index: int = 0,
        target_index: int = -1,
    ) -> Optional[torch.Tensor]:
        if layer_attention is None:
            return None

        if not isinstance(layer_attention, torch.Tensor):
            return None

        attn = layer_attention.detach()

        if attn.numel() == 0:
            return None

        if attn.dim() == 4:
            if batch_index >= int(attn.size(0)):
                return None

            attn = attn[batch_index]

        if attn.dim() == 3:
            # [heads, tgt_len, src_len]
            target_index = self._normalize_index(target_index, int(attn.size(1)))

            if target_index is None:
                return None

            head_vectors = attn[:, target_index, :].float()

            if self.head_strategy.lower() == "max":
                return head_vectors.max(dim=0).values

            return head_vectors.mean(dim=0)

        if attn.dim() == 2:
            # [tgt_len, src_len]
            target_index = self._normalize_index(target_index, int(attn.size(0)))

            if target_index is None:
                return None

            return attn[target_index].float()

        if attn.dim() == 1:
            return attn.float()

        return None

    def _normalize_index(
        self,
        index: int,
        length: int,
    ) -> Optional[int]:
        if length <= 0:
            return None

        index = int(index)

        if index < 0:
            index = length + index

        if index < 0 or index >= length:
            return None

        return index

    def _apply_source_mask(
        self,
        vector: torch.Tensor,
        source_attention_mask: torch.Tensor,
        batch_index: int = 0,
    ) -> Optional[torch.Tensor]:
        mask = source_attention_mask

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, device=vector.device)

        mask = mask.to(device=vector.device)

        if mask.dim() == 2:
            if batch_index >= int(mask.size(0)):
                return vector

            mask = mask[batch_index]

        if mask.dim() != 1:
            return vector

        min_len = min(int(vector.numel()), int(mask.numel()))

        vector = vector[:min_len] * mask[:min_len].float()

        if float(vector.sum().item()) <= 0.0:
            return None

        return vector

    def update(
        self,
        cross_attentions,
        constraints=None,
        covered_spans=None,
    ) -> Dict[str, Any]:
        """
        API tương thích code cũ.
        Code mới dùng get_focus().
        """

        info = self.get_focus(
            cross_attentions=cross_attentions,
            batch_index=0,
            target_index=-1,
            source_attention_mask=None,
        )

        return {
            "attention_available": info.valid,
            "focus_position": info.focus_pos,
            "focus_score": info.focus_score,
            "attention_top": list(zip(info.topk_positions, info.topk_scores)),
            "attention_vector": info.vector.detach().cpu().tolist()
            if info.vector is not None
            else None,
            "attention_debug": info.to_dict(compact=True),
            "attention_triggered": info.valid,
            "active_constraint": None,
        }

    def debug_info(self) -> Dict[str, Any]:
        return dict(self.last_debug)