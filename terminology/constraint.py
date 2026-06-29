from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import copy
import hashlib


# ============================================================
# Constraint Type
# ============================================================

class ConstraintType(str, Enum):
    """
    Loại constraint.

    SOFT:
        Không ép trong decoding.
        Chỉ dùng để validate / debug sau dịch.

    HARD:
        Ép bằng FSA khi attention đi vào source span.

    PROTECTED:
        Thuật ngữ cần giữ nguyên mặt chữ.
        Ví dụ:
            API
            JSON
            logits
            Transformer

        Trong decoding giai đoạn đầu có thể xử lý giống HARD.
    """

    SOFT = "soft"
    HARD = "hard"
    PROTECTED = "protected"


class ConstraintState(str, Enum):
    """
    Trạng thái constraint trong từng hypothesis.

    PENDING:
        Chưa được kích hoạt.

    ACTIVE:
        Đang được sinh bằng FSA.

    DONE:
        Đã sinh xong target phrase.

    BLOCKED:
        Không sử dụng constraint này nữa.
        Dùng cho overlap/conflict nếu cần.
    """

    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    DONE = "DONE"
    BLOCKED = "BLOCKED"


class ConstraintPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# Normalization helpers
# ============================================================

def normalize_space(
    text: Any,
) -> str:
    if text is None:
        return ""

    return " ".join(
        str(
            text
        ).strip().split()
    )


def normalize_source_key(
    text: Any,
) -> str:
    return normalize_space(
        text
    ).lower()


def parse_bool(
    value: Any,
    default: bool = False,
) -> bool:
    if value is None:
        return default

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        int,
    ):
        return value != 0

    text = str(
        value
    ).strip().lower()

    if text in {
        "1",
        "true",
        "yes",
        "y",
        "on",
        "force",
        "forced",
        "protect",
        "protected",
    }:
        return True

    if text in {
        "0",
        "false",
        "no",
        "n",
        "off",
        "",
        "none",
        "null",
    }:
        return False

    return default


def normalize_constraint_type(
    value: Any,
    force: bool = False,
    protect: bool = False,
) -> ConstraintType:
    """
    Chuẩn hóa constraint_type từ CSV/object.

    Quy tắc quan trọng:
        - Nếu CSV ghi rõ constraint_type=soft thì luôn là SOFT,
          dù protect=1.
        - protect=1 chỉ có nghĩa là cần bảo vệ/hiển thị,
          không tự biến SOFT thành PROTECTED.
        - Chỉ khi không có constraint_type rõ ràng mới suy luận
          từ protect/force để tương thích CSV cũ.
    """

    if isinstance(value, ConstraintType):
        return value

    if hasattr(value, "value") and str(getattr(value, "value", "")).lower() in {
        "soft",
        "hard",
        "protected",
    }:
        return ConstraintType(str(value.value).lower())

    if value is None:
        if protect:
            return ConstraintType.PROTECTED

        if force:
            return ConstraintType.HARD

        return ConstraintType.SOFT

    text = str(
        value
    ).strip().lower().replace(
        "-",
        "_",
    )

    if text.startswith("constrainttype."):
        text = text.split(".", 1)[1]

    aliases = {
        "soft": ConstraintType.SOFT,
        "soft_constraint": ConstraintType.SOFT,
        "soft_term": ConstraintType.SOFT,

        "hard": ConstraintType.HARD,
        "force": ConstraintType.HARD,
        "forced": ConstraintType.HARD,
        "relaxed": ConstraintType.HARD,
        "relaxed_hard": ConstraintType.HARD,
        "strict": ConstraintType.HARD,
        "strict_hard": ConstraintType.HARD,

        "copy": ConstraintType.PROTECTED,
        "protected": ConstraintType.PROTECTED,
        "protected_copy": ConstraintType.PROTECTED,
        "placeholder": ConstraintType.PROTECTED,
    }

    if text in aliases:
        return aliases[
            text
        ]

    if protect:
        return ConstraintType.PROTECTED

    if force:
        return ConstraintType.HARD

    return ConstraintType.SOFT


def normalize_priority(
    value: Any,
) -> ConstraintPriority:
    if value is None:
        return ConstraintPriority.MEDIUM

    text = str(
        value
    ).strip().lower()

    if text in {
        "critical",
        "strict",
    }:
        return ConstraintPriority.CRITICAL

    if text == "high":
        return ConstraintPriority.HIGH

    if text == "low":
        return ConstraintPriority.LOW

    return ConstraintPriority.MEDIUM


def priority_score(
    priority: Any,
) -> int:
    priority = normalize_priority(
        priority
    )

    return {
        ConstraintPriority.LOW: 1,
        ConstraintPriority.MEDIUM: 2,
        ConstraintPriority.HIGH: 3,
        ConstraintPriority.CRITICAL: 4,
    }.get(
        priority,
        2,
    )


# ============================================================
# Constraint
# ============================================================

@dataclass
class Constraint:
    """
    Một ràng buộc thuật ngữ cho một câu nguồn.

    Object này được truyền qua pipeline:

        detector
        -> fsa_builder
        -> attention_activator
        -> logits_masker
        -> beam_search
        -> validator

    Quan trọng:
        Constraint không phải global state.
        Mỗi hypothesis trong beam phải giữ bản clone riêng.

    Quy ước span:
        word_span  = [start, end)
        token_span = [start, end)
        char_span  = [start, end)
    """

    id: str

    source_phrase: str
    target_phrase: str

    category: str = "general"
    priority: ConstraintPriority = ConstraintPriority.MEDIUM
    constraint_type: ConstraintType = ConstraintType.SOFT

    force: bool = False
    protect: bool = False

    word_span: Optional[Tuple[int, int]] = None
    token_span: Optional[Tuple[int, int]] = None
    char_span: Optional[Tuple[int, int]] = None

    source_order: int = 0

    target_token_ids: List[int] = field(
        default_factory=list
    )
    target_tokens: List[str] = field(
        default_factory=list
    )

    state: ConstraintState = ConstraintState.PENDING
    fsa: Optional[Any] = None

    covered: bool = False

    alternatives: List[str] = field(
        default_factory=list
    )

    meta: Dict[str, Any] = field(
        default_factory=dict
    )

    # --------------------------------------------------------
    # Factory
    # --------------------------------------------------------

    @staticmethod
    def make_id(
        source_phrase: str,
        target_phrase: str,
        word_span: Optional[Tuple[int, int]] = None,
        token_span: Optional[Tuple[int, int]] = None,
    ) -> str:
        raw = "|".join(
            [
                normalize_source_key(
                    source_phrase
                ),
                normalize_space(
                    target_phrase
                ),
                str(
                    word_span
                ),
                str(
                    token_span
                ),
            ]
        )

        return hashlib.md5(
            raw.encode(
                "utf-8"
            )
        ).hexdigest()[
            :12
        ]

    @classmethod
    def create(
        cls,
        source_phrase: str,
        target_phrase: str,
        category: str = "general",
        priority: Any = "medium",
        constraint_type: Any = None,
        force: Any = False,
        protect: Any = False,
        word_span: Optional[Tuple[int, int]] = None,
        token_span: Optional[Tuple[int, int]] = None,
        char_span: Optional[Tuple[int, int]] = None,
        source_order: int = 0,
        alternatives: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "Constraint":
        source_phrase = normalize_space(
            source_phrase
        )

        target_phrase = normalize_space(
            target_phrase
        )

        protect_bool = parse_bool(
            protect,
            default=False,
        )

        force_bool = parse_bool(
            force,
            default=False,
        )

        ctype = normalize_constraint_type(
            constraint_type,
            force=force_bool,
            protect=protect_bool,
        )

        # force phải lấy từ CSV / runtime policy.
        # Không tự biến hard/protected thành force=True ở đây.
        if ctype == ConstraintType.PROTECTED:
            protect_bool = True

        cid = cls.make_id(
            source_phrase=source_phrase,
            target_phrase=target_phrase,
            word_span=word_span,
            token_span=token_span,
        )

        return cls(
            id=cid,
            source_phrase=source_phrase,
            target_phrase=target_phrase,
            category=normalize_space(
                category
            )
            or "general",
            priority=normalize_priority(
                priority
            ),
            constraint_type=ctype,
            force=force_bool,
            protect=protect_bool,
            word_span=word_span,
            token_span=token_span,
            char_span=char_span,
            source_order=source_order,
            alternatives=list(
                alternatives or []
            ),
            meta=dict(
                meta or {}
            ),
        )

    # --------------------------------------------------------
    # Type checks
    # --------------------------------------------------------

    def is_soft(
        self,
    ) -> bool:
        return self.constraint_type == ConstraintType.SOFT

    def is_hard(
        self,
    ) -> bool:
        return self.constraint_type == ConstraintType.HARD

    def is_protected(
        self,
    ) -> bool:
        return self.constraint_type == ConstraintType.PROTECTED

    def should_force(
        self,
    ) -> bool:
        return bool(
            self.force
        ) and self.constraint_type in {
            ConstraintType.HARD,
            ConstraintType.PROTECTED,
        }

    # --------------------------------------------------------
    # State checks
    # --------------------------------------------------------

    def is_pending(
        self,
    ) -> bool:
        return self.state == ConstraintState.PENDING

    def is_active(
        self,
    ) -> bool:
        return self.state == ConstraintState.ACTIVE

    def is_done(
        self,
    ) -> bool:
        return self.state == ConstraintState.DONE

    def is_blocked(
        self,
    ) -> bool:
        return self.state == ConstraintState.BLOCKED

    # --------------------------------------------------------
    # State transitions
    # --------------------------------------------------------

    def activate(
        self,
    ) -> None:
        if self.is_done() or self.is_blocked():
            return

        self.state = ConstraintState.ACTIVE

        if self.fsa is not None and hasattr(
            self.fsa,
            "activate",
        ):
            self.fsa.activate()

    def mark_done(
        self,
    ) -> None:
        self.state = ConstraintState.DONE
        self.covered = True

        if self.fsa is not None and hasattr(
            self.fsa,
            "force_done",
        ):
            self.fsa.force_done()

    def block(
        self,
    ) -> None:
        if self.is_done():
            return

        self.state = ConstraintState.BLOCKED

    def reset_runtime(
        self,
    ) -> None:
        self.state = ConstraintState.PENDING
        self.covered = False

        if self.fsa is not None and hasattr(
            self.fsa,
            "reset",
        ):
            self.fsa.reset()

    # --------------------------------------------------------
    # Span helpers
    # --------------------------------------------------------

    def has_word_span(
        self,
    ) -> bool:
        if self.word_span is None:
            return False

        start, end = self.word_span

        return start is not None and end is not None and int(
            end
        ) > int(
            start
        )

    def has_token_span(
        self,
    ) -> bool:
        if self.token_span is None:
            return False

        start, end = self.token_span

        return start is not None and end is not None and int(
            end
        ) > int(
            start
        )

    def contains_source_token(
        self,
        source_pos: int,
    ) -> bool:
        if not self.has_token_span():
            return False

        start, end = self.token_span

        return int(
            start
        ) <= int(
            source_pos
        ) < int(
            end
        )

    def span_attention_score(
        self,
        attention_vector,
    ) -> float:
        """
        Tính tổng attention nằm trên source span của constraint.

        attention_vector:
            list[float]
            hoặc torch.Tensor shape [src_len]
        """

        if attention_vector is None:
            return 0.0

        if not self.has_token_span():
            return 0.0

        start, end = self.token_span

        try:
            span_attn = attention_vector[
                int(
                    start
                ) : int(
                    end
                )
            ]

            if hasattr(
                span_attn,
                "sum",
            ):
                value = span_attn.sum()

                return float(
                    value.item()
                    if hasattr(
                        value,
                        "item",
                    )
                    else value
                )

            return float(
                sum(
                    span_attn
                )
            )

        except Exception:
            return 0.0

    # --------------------------------------------------------
    # FSA helpers
    # --------------------------------------------------------

    def next_token_id(
        self,
    ) -> Optional[int]:
        if self.fsa is None:
            return None

        if not hasattr(
            self.fsa,
            "next_token_id",
        ):
            return None

        return self.fsa.next_token_id()

    def allowed_token_ids(
        self,
    ) -> List[int]:
        if self.fsa is None:
            return []

        if hasattr(
            self.fsa,
            "allowed_token_ids",
        ):
            return [
                int(
                    token_id
                )
                for token_id in self.fsa.allowed_token_ids()
            ]

        nxt = self.next_token_id()

        if nxt is None:
            return []

        return [
            int(
                nxt
            )
        ]

    def fsa_position(
        self,
    ) -> int:
        if self.fsa is None:
            return 0

        return int(
            getattr(
                self.fsa,
                "position",
                0,
            )
        )

    def fsa_length(
        self,
    ) -> int:
        if self.fsa is None:
            return 0

        return int(
            getattr(
                self.fsa,
                "length",
                0,
            )
        )

    def fsa_is_done(
        self,
    ) -> bool:
        if self.fsa is None:
            return False

        if hasattr(
            self.fsa,
            "is_done",
        ):
            return bool(
                self.fsa.is_done()
            )

        return self.fsa_position() >= self.fsa_length() > 0

    # --------------------------------------------------------
    # Clone / serialization
    # --------------------------------------------------------

    def clone(
        self,
    ) -> "Constraint":
        return copy.deepcopy(
            self
        )

    def to_dict(
        self,
        include_runtime: bool = True,
    ) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "source_phrase": self.source_phrase,
            "target_phrase": self.target_phrase,
            "category": self.category,
            "priority": self.priority.value
            if isinstance(
                self.priority,
                ConstraintPriority,
            )
            else str(
                self.priority
            ),
            "constraint_type": self.constraint_type.value
            if isinstance(
                self.constraint_type,
                ConstraintType,
            )
            else str(
                self.constraint_type
            ),
            "force": self.force,
            "protect": self.protect,
            "word_span": list(
                self.word_span
            )
            if self.word_span is not None
            else None,
            "token_span": list(
                self.token_span
            )
            if self.token_span is not None
            else None,
            "char_span": list(
                self.char_span
            )
            if self.char_span is not None
            else None,
            "source_order": self.source_order,
            "target_token_ids": list(
                self.target_token_ids
            ),
            "target_tokens": list(
                self.target_tokens
            ),
            "alternatives": list(
                self.alternatives
            ),
        }

        if include_runtime:
            data.update(
                {
                    "state": self.state.value
                    if isinstance(
                        self.state,
                        ConstraintState,
                    )
                    else str(
                        self.state
                    ),
                    "covered": self.covered,
                    "has_fsa": self.fsa is not None,
                    "fsa_position": self.fsa_position(),
                    "fsa_length": self.fsa_length(),
                    "fsa_done": self.fsa_is_done(),
                    "display_status": self.display_status(),
                }
            )

        return data

    def display_status(
        self,
    ) -> str:
        if self.is_soft():
            return "SOFT"

        if self.is_protected():
            if self.is_done():
                return "PROTECTED_DONE"

            if self.is_active():
                return "PROTECTED_ACTIVE"

            return "PROTECTED_PENDING"

        if self.is_done():
            return "DONE"

        if self.is_active():
            return "ACTIVE"

        if self.is_blocked():
            return "BLOCKED"

        return "PENDING"


# ============================================================
# Utility functions for constraint lists
# ============================================================

def clone_constraints(
    constraints: List[Constraint],
) -> List[Constraint]:
    return [
        constraint.clone()
        for constraint in constraints or []
    ]


def hard_constraints(
    constraints: List[Constraint],
) -> List[Constraint]:
    return [
        constraint
        for constraint in constraints or []
        if constraint.should_force()
    ]


def soft_constraints(
    constraints: List[Constraint],
) -> List[Constraint]:
    return [
        constraint
        for constraint in constraints or []
        if constraint.is_soft()
    ]


def pending_constraints(
    constraints: List[Constraint],
) -> List[Constraint]:
    return [
        constraint
        for constraint in constraints or []
        if constraint.is_pending()
    ]


def active_constraints(
    constraints: List[Constraint],
) -> List[Constraint]:
    return [
        constraint
        for constraint in constraints or []
        if constraint.is_active()
    ]


def done_constraints(
    constraints: List[Constraint],
) -> List[Constraint]:
    return [
        constraint
        for constraint in constraints or []
        if constraint.is_done()
    ]


def all_forced_done(
    constraints: List[Constraint],
) -> bool:
    forced = hard_constraints(
        constraints
    )

    if not forced:
        return True

    return all(
        constraint.is_done()
        for constraint in forced
    )


def constraint_progress_key(
    constraints: List[Constraint],
) -> Tuple[int, int, Tuple[int, ...]]:
    """
    Key dùng cho multi-stack / DBA.

    Return:
        (
            số forced constraint DONE,
            có ACTIVE constraint hay không,
            tuple fsa positions
        )
    """

    forced = hard_constraints(
        constraints
    )

    done_count = sum(
        1
        for constraint in forced
        if constraint.is_done()
    )

    has_active = 1 if any(
        constraint.is_active()
        for constraint in forced
    ) else 0

    positions = tuple(
        constraint.fsa_position()
        for constraint in forced
    )

    return (
        done_count,
        has_active,
        positions,
    )