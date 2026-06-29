from __future__ import annotations

from typing import Any, Dict, List, Optional
import re
import unicodedata

from terminology.constraint import (
    Constraint,
    ConstraintType,
)


class ConstraintValidator:
    """
    Validator cuối pipeline.

    Vai trò:

        translation + constraints
        -> kiểm tra thuật ngữ có xuất hiện không
        -> trả dict cho UI

    File này KHÔNG:
        - sửa translation
        - post-edit thuật ngữ
        - activate constraint
        - mask logits
        - gọi FSA.step()
        - ép thêm target phrase vào output

    Chính sách:

        SOFT:
            Không bắt buộc.
            Nếu target xuất hiện thì counted satisfied.
            Nếu không xuất hiện thì không làm ok=False.

        HARD:
            Bắt buộc target xuất hiện hoặc FSA/state DONE.

        PROTECTED:
            Bắt buộc xuất hiện đúng mặt chữ nếu protected_exact_match=True.
    """

    def __init__(
        self,
        protected_exact_match: bool = True,
    ):
        self.protected_exact_match = bool(
            protected_exact_match
        )

    # --------------------------------------------------
    # Text normalize
    # --------------------------------------------------

    def _normalize_unicode(
        self,
        text: Any,
    ) -> str:
        if text is None:
            return ""

        return unicodedata.normalize(
            "NFC",
            str(
                text
            ),
        )

    def _normalize_space(
        self,
        text: Any,
    ) -> str:
        text = self._normalize_unicode(
            text
        )

        text = re.sub(
            r"[\u200b\u200c\u200d\ufeff]",
            "",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    def _normalize_for_soft_match(
        self,
        text: Any,
    ) -> str:
        return self._normalize_space(
            text
        ).lower()

    def _contains_normalized(
        self,
        translation: str,
        target_phrase: str,
    ) -> bool:
        target = self._normalize_for_soft_match(
            target_phrase
        )

        output = self._normalize_for_soft_match(
            translation
        )

        if not target or not output:
            return False

        return target in output

    def _contains_exact_surface(
        self,
        translation: str,
        target_phrase: str,
    ) -> bool:
        """
        Match đúng mặt chữ.

        Dùng cho protected terms:
            API
            JSON
            Transformer
            logits

        Có normalize Unicode và whitespace,
        nhưng KHÔNG lowercase.
        """

        target = self._normalize_space(
            target_phrase
        )

        output = self._normalize_space(
            translation
        )

        if not target or not output:
            return False

        return target in output

    # --------------------------------------------------
    # Constraint helpers
    # --------------------------------------------------

    def _constraint_type_value(
        self,
        constraint: Constraint,
    ) -> str:
        value = getattr(
            constraint,
            "constraint_type",
            ConstraintType.SOFT,
        )

        if hasattr(
            value,
            "value",
        ):
            return str(
                value.value
            ).lower()

        return str(
            value
        ).lower()

    def _state_value(
        self,
        constraint: Constraint,
    ) -> str:
        value = getattr(
            constraint,
            "state",
            "",
        )

        if hasattr(
            value,
            "value",
        ):
            return str(
                value.value
            )

        return str(
            value
        )

    def _is_soft(
        self,
        constraint: Constraint,
    ) -> bool:
        return self._constraint_type_value(
            constraint
        ) == "soft"

    def _is_hard(
        self,
        constraint: Constraint,
    ) -> bool:
        return self._constraint_type_value(
            constraint
        ) == "hard"

    def _is_protected(
        self,
        constraint: Constraint,
    ) -> bool:
        return self._constraint_type_value(
            constraint
        ) == "protected"

    def _is_required(
        self,
        constraint: Constraint,
    ) -> bool:
        """
        Required nghĩa là constraint bắt buộc phải được thỏa.

        HARD và PROTECTED là required.
        SOFT không required.
        """

        if self._is_hard(
            constraint
        ):
            return True

        if self._is_protected(
            constraint
        ):
            return True

        if bool(
            getattr(
                constraint,
                "force",
                False,
            )
        ) and getattr(
            constraint,
            "fsa",
            None,
        ) is not None:
            return True

        return False

    def _fsa_done(
        self,
        constraint: Constraint,
    ) -> bool:
        fsa = getattr(
            constraint,
            "fsa",
            None,
        )

        if fsa is None:
            return False

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

    def _state_done(
        self,
        constraint: Constraint,
    ) -> bool:
        if hasattr(
            constraint,
            "is_done",
        ):
            try:
                if constraint.is_done():
                    return True
            except Exception:
                pass

        return self._state_value(
            constraint
        ).upper() == "DONE"

    def _target_found(
        self,
        translation: str,
        constraint: Constraint,
    ) -> bool:
        target_phrase = getattr(
            constraint,
            "target_phrase",
            "",
        )

        if not target_phrase:
            return False

        if self._is_protected(
            constraint
        ) and self.protected_exact_match:
            return self._contains_exact_surface(
                translation,
                target_phrase,
            )

        return self._contains_normalized(
            translation,
            target_phrase,
        )

    # --------------------------------------------------
    # Validate one
    # --------------------------------------------------

    def _validate_one(
        self,
        translation: str,
        constraint: Constraint,
    ) -> Dict[str, Any]:
        source_phrase = getattr(
            constraint,
            "source_phrase",
            "",
        )

        target_phrase = getattr(
            constraint,
            "target_phrase",
            "",
        )

        constraint_type = self._constraint_type_value(
            constraint
        )

        force = bool(
            getattr(
                constraint,
                "force",
                False,
            )
        )

        protect = bool(
            getattr(
                constraint,
                "protect",
                False,
            )
        )

        required = self._is_required(
            constraint
        )

        lexical_found = self._target_found(
            translation,
            constraint,
        )

        fsa_done = self._fsa_done(
            constraint
        )

        state_done = self._state_done(
            constraint
        )

        hard_like = required
        soft = not required

        if hard_like:
            satisfied = bool(
                lexical_found or fsa_done or state_done
            )
        else:
            satisfied = bool(
                lexical_found
            )

        if hard_like and satisfied:
            reason = "required_satisfied"
        elif hard_like and not satisfied:
            reason = "required_missing"
        elif soft and lexical_found:
            reason = "soft_found"
        else:
            reason = "soft_not_required"

        return {
            "id": getattr(
                constraint,
                "id",
                None,
            ),
            "source_phrase": source_phrase,
            "target_phrase": target_phrase,

            "constraint_type": constraint_type,
            "force": force,
            "protect": protect,
            "required": required,

            "satisfied": satisfied,
            "missing": not satisfied if hard_like else False,
            "lexical_found": lexical_found,
            "fsa_done": fsa_done,
            "state_done": state_done,
            "hard_like": hard_like,
            "soft": soft,

            "state": self._state_value(
                constraint
            ),
            "reason": reason,

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

            "target_token_ids": list(
                getattr(
                    constraint,
                    "target_token_ids",
                    [],
                )
            ),
            "target_tokens": list(
                getattr(
                    constraint,
                    "target_tokens",
                    [],
                )
            ),
        }

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def validate(
        self,
        translation: str,
        constraints: Optional[List[Constraint]],
    ) -> Dict[str, Any]:
        constraints = list(
            constraints or []
        )

        details = []

        for constraint in constraints:
            if constraint is None:
                continue

            details.append(
                self._validate_one(
                    translation=translation,
                    constraint=constraint,
                )
            )

        total = len(
            details
        )

        hard_items = [
            item
            for item in details
            if item["hard_like"]
        ]

        soft_items = [
            item
            for item in details
            if item["soft"]
        ]

        satisfied_items = [
            item
            for item in details
            if item["satisfied"]
        ]

        hard_satisfied_items = [
            item
            for item in hard_items
            if item["satisfied"]
        ]

        soft_satisfied_items = [
            item
            for item in soft_items
            if item["satisfied"]
        ]

        missing_required_items = [
            item
            for item in hard_items
            if not item["satisfied"]
        ]

        hard_total = len(
            hard_items
        )

        soft_total = len(
            soft_items
        )

        satisfied = len(
            satisfied_items
        )

        missing = len(
            missing_required_items
        )

        hard_satisfied = len(
            hard_satisfied_items
        )

        soft_satisfied = len(
            soft_satisfied_items
        )

        hard_missing = len(
            missing_required_items
        )

        soft_missing = max(
            0,
            soft_total - soft_satisfied,
        )

        coverage = (
            satisfied / total
            if total > 0
            else 1.0
        )

        hard_coverage = (
            hard_satisfied / hard_total
            if hard_total > 0
            else 1.0
        )

        soft_coverage = (
            soft_satisfied / soft_total
            if soft_total > 0
            else 1.0
        )

        lexical_found = sum(
            1
            for item in details
            if item["lexical_found"]
        )

        fsa_done = sum(
            1
            for item in details
            if item["fsa_done"]
        )

        state_done = sum(
            1
            for item in details
            if item["state_done"]
        )

        ok = hard_missing == 0

        return {
            "ok": ok,

            # UI-compatible summary
            "total": total,
            "satisfied": satisfied,
            "missing": missing,
            "coverage": coverage,

            "hard_total": hard_total,
            "hard_satisfied": hard_satisfied,
            "hard_missing": hard_missing,
            "hard_coverage": hard_coverage,

            "soft_total": soft_total,
            "soft_satisfied": soft_satisfied,
            "soft_missing": soft_missing,
            "soft_coverage": soft_coverage,

            "lexical_found": lexical_found,
            "fsa_done": fsa_done,
            "state_done": state_done,

            # Compatibility với tên cũ nếu UI/code cũ dùng
            "required_total": hard_total,
            "passed_total": satisfied,
            "failed_total": hard_missing,
            "missing_required": missing_required_items,

            "missing_constraints": missing_required_items,
            "hard_missing_constraints": missing_required_items,
            "soft_missing_constraints": [
                item
                for item in soft_items
                if not item["satisfied"]
            ],

            "details": details,
            "items": details,
        }

    def validate_to_dict(
        self,
        translation: str,
        constraints: Optional[List[Constraint]],
    ) -> Dict[str, Any]:
        return self.validate(
            translation=translation,
            constraints=constraints,
        )

    def __call__(
        self,
        translation: str,
        constraints: Optional[List[Constraint]],
    ) -> Dict[str, Any]:
        return self.validate(
            translation=translation,
            constraints=constraints,
        )