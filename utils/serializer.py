from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    import torch
except Exception:
    torch = None


# ============================================================
# Generic safe serializer
# ============================================================

def safe_serialize(
    value: Any,
    max_depth: int = 5,
    _depth: int = 0,
) -> Any:
    """
    Chuyển object bất kỳ thành dạng an toàn cho UI/JSON.

    Hỗ trợ:
        - primitive
        - list / tuple / set
        - dict
        - Enum
        - dataclass
        - torch.Tensor
        - object có to_dict()
        - object thường có __dict__

    Mục tiêu:
        Không để UI bị lỗi vì object không serialize được.
    """

    if _depth > max_depth:
        return str(
            value
        )

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        Enum,
    ):
        return value.value

    if torch is not None and isinstance(
        value,
        torch.Tensor,
    ):
        try:
            if value.numel() == 1:
                return value.detach().cpu().item()

            return value.detach().cpu().tolist()
        except Exception:
            return str(
                value
            )

    if isinstance(
        value,
        dict,
    ):
        return {
            str(
                safe_serialize(
                    key,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                )
            ): safe_serialize(
                item,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            safe_serialize(
                item,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            for item in value
        ]

    if hasattr(
        value,
        "to_dict",
    ):
        try:
            return safe_serialize(
                value.to_dict(),
                max_depth=max_depth,
                _depth=_depth + 1,
            )
        except TypeError:
            try:
                return safe_serialize(
                    value.to_dict(
                        include_runtime=True
                    ),
                    max_depth=max_depth,
                    _depth=_depth + 1,
                )
            except Exception:
                pass
        except Exception:
            pass

    if is_dataclass(
        value
    ):
        try:
            return safe_serialize(
                asdict(
                    value
                ),
                max_depth=max_depth,
                _depth=_depth + 1,
            )
        except Exception:
            pass

    if hasattr(
        value,
        "__dict__",
    ):
        try:
            data = {}

            for key, item in vars(
                value
            ).items():
                if key.startswith(
                    "_"
                ):
                    continue

                data[
                    key
                ] = safe_serialize(
                    item,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                )

            return data
        except Exception:
            pass

    return str(
        value
    )


# ============================================================
# Constraint serializer
# ============================================================

def constraint_to_dict(
    constraint: Any,
) -> Dict[str, Any]:
    """
    Serialize một Constraint.

    Ưu tiên dùng constraint.to_dict().
    Nếu không có thì fallback thủ công.
    """

    if constraint is None:
        return {}

    if hasattr(
        constraint,
        "to_dict",
    ):
        try:
            return safe_serialize(
                constraint.to_dict(
                    include_runtime=True
                )
            )
        except TypeError:
            try:
                return safe_serialize(
                    constraint.to_dict()
                )
            except Exception:
                pass
        except Exception:
            pass

    fsa = getattr(
        constraint,
        "fsa",
        None,
    )

    if fsa is not None and hasattr(
        fsa,
        "to_dict",
    ):
        try:
            fsa_data = fsa.to_dict()
        except Exception:
            fsa_data = str(
                fsa
            )
    else:
        fsa_data = None

    constraint_type = getattr(
        constraint,
        "constraint_type",
        None,
    )

    if hasattr(
        constraint_type,
        "value",
    ):
        constraint_type = constraint_type.value

    state = getattr(
        constraint,
        "state",
        None,
    )

    if hasattr(
        state,
        "value",
    ):
        state = state.value

    return safe_serialize(
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
            "category": getattr(
                constraint,
                "category",
                None,
            ),
            "priority": getattr(
                constraint,
                "priority",
                None,
            ),
            "constraint_type": constraint_type,
            "force": getattr(
                constraint,
                "force",
                None,
            ),
            "protect": getattr(
                constraint,
                "protect",
                None,
            ),
            "word_span": getattr(
                constraint,
                "word_span",
                None,
            ),
            "token_span": getattr(
                constraint,
                "token_span",
                None,
            ),
            "char_span": getattr(
                constraint,
                "char_span",
                None,
            ),
            "source_order": getattr(
                constraint,
                "source_order",
                None,
            ),
            "state": state,
            "covered": getattr(
                constraint,
                "covered",
                None,
            ),
            "target_token_ids": getattr(
                constraint,
                "target_token_ids",
                [],
            ),
            "target_tokens": getattr(
                constraint,
                "target_tokens",
                [],
            ),
            "has_fsa": fsa is not None,
            "fsa": fsa_data,
            "meta": getattr(
                constraint,
                "meta",
                {},
            ),
        }
    )


def constraints_to_dict(
    constraints: Optional[List[Any]],
) -> List[Dict[str, Any]]:
    if not constraints:
        return []

    return [
        constraint_to_dict(
            constraint
        )
        for constraint in constraints
    ]


# ============================================================
# Trace serializer
# ============================================================

def trace_item_to_dict(
    item: Any,
) -> Dict[str, Any]:
    """
    Serialize một trace item.

    Trace mới thường đã là dict.
    Nhưng bên trong có thể còn Constraint/Tensor/Enum.
    """

    serialized = safe_serialize(
        item,
        max_depth=6,
    )

    if isinstance(
        serialized,
        dict,
    ):
        return serialized

    return {
        "value": serialized
    }


def trace_to_dict(
    trace: Optional[List[Any]],
) -> List[Dict[str, Any]]:
    if not trace:
        return []

    return [
        trace_item_to_dict(
            item
        )
        for item in trace
    ]


# ============================================================
# Validation serializer
# ============================================================

def validation_to_dict(
    validation: Any,
) -> Dict[str, Any]:
    """
    Serialize kết quả ConstraintValidator.

    Hỗ trợ:
        - ConstraintValidationResult mới có .to_dict()
        - dict cũ
        - object thường
    """

    if validation is None:
        return {}

    if isinstance(
        validation,
        dict,
    ):
        return safe_serialize(
            validation
        )

    if hasattr(
        validation,
        "to_dict",
    ):
        try:
            result = validation.to_dict()

            if isinstance(
                result,
                dict,
            ):
                return safe_serialize(
                    result
                )

            return {
                "value": safe_serialize(
                    result
                )
            }
        except Exception:
            pass

    serialized = safe_serialize(
        validation
    )

    if isinstance(
        serialized,
        dict,
    ):
        return serialized

    return {
        "value": serialized
    }


# ============================================================
# Beam / result helpers
# ============================================================

def beam_summary_to_dict(
    beam_summary: Any,
) -> Dict[str, Any]:
    if beam_summary is None:
        return {}

    serialized = safe_serialize(
        beam_summary
    )

    if isinstance(
        serialized,
        dict,
    ):
        return serialized

    return {
        "value": serialized
    }


def translation_result_to_dict(
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Serialize toàn bộ result từ Translator.translate().
    Dùng khi cần export/debug.
    """

    if result is None:
        return {}

    output = {}

    for key, value in result.items():
        if key == "constraints":
            output[
                key
            ] = constraints_to_dict(
                value
            )
        elif key == "trace":
            output[
                key
            ] = trace_to_dict(
                value
            )
        elif key == "constraint_validation":
            output[
                key
            ] = validation_to_dict(
                value
            )
        else:
            output[
                key
            ] = safe_serialize(
                value,
                max_depth=6,
            )

    return output