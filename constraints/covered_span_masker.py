from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch


class CoveredSpanMasker:
    """
    Che các source span đã dịch xong.

    Mục tiêu:
        Nếu constraint đã DONE / covered=True,
        các token nguồn thuộc span đó sẽ bị đưa attention_mask về 0.

    Ví dụ:
        source: The API sends request to server
        server span = [5, 6)

        Sau khi server -> máy chủ DONE:
            attention_mask[5] = 0

    Tác dụng:
        model không còn chú ý lại vào "server",
        giảm lỗi lặp như:
            máy máy chủ
            Cơ sở truy vấn cơ sở dữ liệu
    """

    def __init__(
        self,
        keep_at_least_one_token: bool = True,
        enable_source_mask: bool = True,
    ):
        self.keep_at_least_one_token = bool(
            keep_at_least_one_token
        )

        self.enable_source_mask = bool(
            enable_source_mask
        )

        self.last_debug: Dict[str, Any] = {
            "enabled": self.enable_source_mask,
            "masked_spans": [],
            "reason": "init",
        }

    # --------------------------------------------------
    # State helpers
    # --------------------------------------------------

    def _state_text(
        self,
        constraint,
    ) -> str:
        state = getattr(
            constraint,
            "state",
            "",
        )

        if hasattr(
            state,
            "value",
        ):
            state = state.value

        return str(
            state
        ).lower()

    def _is_done(
        self,
        constraint,
    ) -> bool:
        if constraint is None:
            return False

        if bool(
            getattr(
                constraint,
                "covered",
                False,
            )
        ):
            return True

        if hasattr(
            constraint,
            "is_done",
        ):
            try:
                if constraint.is_done():
                    return True
            except Exception:
                pass

        state_text = self._state_text(
            constraint
        )

        if state_text.endswith(
            "done"
        ):
            return True

        if state_text == "done":
            return True

        fsa = getattr(
            constraint,
            "fsa",
            None,
        )

        if fsa is not None:
            value = getattr(
                fsa,
                "is_done",
                False,
            )

            if callable(
                value
            ):
                try:
                    return bool(
                        value()
                    )
                except Exception:
                    return False

            return bool(
                value
            )

        return False

    def _span_from_constraint(
        self,
        constraint,
    ) -> Optional[Tuple[int, int]]:
        if constraint is None:
            return None

        span = getattr(
            constraint,
            "token_span",
            None,
        )

        if span is None:
            span = getattr(
                constraint,
                "word_span",
                None,
            )

        if span is None:
            return None

        try:
            start = int(
                span[
                    0
                ]
            )

            end = int(
                span[
                    1
                ]
            )
        except Exception:
            return None

        if end <= start:
            return None

        return start, end

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def mark_covered(
        self,
        constraints: List[Any],
    ) -> List[Any]:
        """
        Đánh dấu covered=True cho constraint đã DONE.

        Hàm này không tự quyết định dịch đúng hay sai.
        Nó chỉ đồng bộ:
            DONE -> covered=True
        """

        for constraint in constraints or []:
            if constraint is None:
                continue

            if self._is_done(
                constraint
            ):
                try:
                    constraint.covered = True
                except Exception:
                    pass

        return constraints

    def apply(
        self,
        attention_mask: Optional[torch.Tensor],
        constraints: Optional[List[Any]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Trả về attention_mask mới sau khi che các span đã covered.

        Shape hỗ trợ:
            [src_len]
            [1, src_len]
        """

        self.last_debug = {
            "enabled": self.enable_source_mask,
            "masked_spans": [],
            "reason": "start",
        }

        if attention_mask is None:
            self.last_debug[
                "reason"
            ] = "attention_mask_none"
            return None

        if not self.enable_source_mask:
            self.last_debug[
                "reason"
            ] = "source_mask_disabled"
            return attention_mask

        if not constraints:
            self.last_debug[
                "reason"
            ] = "no_constraints"
            return attention_mask

        new_mask = attention_mask.clone()

        original_dim = new_mask.dim()

        if original_dim == 1:
            work_mask = new_mask.unsqueeze(
                0
            )
        else:
            work_mask = new_mask

        if work_mask.dim() != 2:
            self.last_debug[
                "reason"
            ] = f"unsupported_mask_dim_{work_mask.dim()}"
            return attention_mask

        src_len = int(
            work_mask.size(
                -1
            )
        )

        masked_spans = []

        for constraint in constraints or []:
            if constraint is None:
                continue

            if not self._is_done(
                constraint
            ):
                continue

            span = self._span_from_constraint(
                constraint
            )

            if span is None:
                continue

            start, end = span

            start = max(
                0,
                min(
                    start,
                    src_len,
                ),
            )

            end = max(
                0,
                min(
                    end,
                    src_len,
                ),
            )

            if end <= start:
                continue

            work_mask[
                :,
                start:end,
            ] = 0

            masked_spans.append(
                {
                    "id": getattr(
                        constraint,
                        "id",
                        None,
                    ),
                    "source_phrase": getattr(
                        constraint,
                        "source_phrase",
                        None,
                    ),
                    "target_phrase": getattr(
                        constraint,
                        "target_phrase",
                        None,
                    ),
                    "span": [
                        start,
                        end,
                    ],
                    "state": self._state_text(
                        constraint
                    ),
                    "covered": bool(
                        getattr(
                            constraint,
                            "covered",
                            False,
                        )
                    ),
                }
            )

        # Tránh mask toàn bộ source.
        if self.keep_at_least_one_token:
            for row in range(
                work_mask.size(
                    0
                )
            ):
                if int(
                    work_mask[
                        row
                    ].sum().item()
                ) == 0:
                    work_mask[
                        row,
                        :
                    ] = attention_mask[
                        row,
                        :
                    ] if attention_mask.dim() == 2 else attention_mask[
                        :
                    ]

                    self.last_debug[
                        "reason"
                    ] = "restored_because_all_tokens_masked"

                    if original_dim == 1:
                        return work_mask.squeeze(
                            0
                        )

                    return work_mask

        self.last_debug = {
            "enabled": self.enable_source_mask,
            "masked_spans": masked_spans,
            "masked_count": len(
                masked_spans
            ),
            "reason": "ok",
        }

        if original_dim == 1:
            return work_mask.squeeze(
                0
            )

        return work_mask

    def debug_info(
        self,
    ) -> Dict[str, Any]:
        return dict(
            self.last_debug
        )