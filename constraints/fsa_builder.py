from __future__ import annotations

from typing import Any, Dict, List, Optional

from constraints.fsa import ConstraintFSA
from terminology.constraint import (
    Constraint,
    ConstraintType,
    normalize_constraint_type,
    normalize_space,
    parse_bool,
)


class FSABuilder:
    """
    Build FSA cho các constraint đã được detector phát hiện.

    Vai trò duy nhất:

        Constraint.target_phrase
        -> tokenizer.encode(...)
        -> target_token_ids
        -> ConstraintFSA
        -> gắn vào Constraint

    File này KHÔNG:
        - quyết định decoder/server/dataset là hard hay soft
        - hard-code danh sách thuật ngữ
        - tự activate constraint
        - đọc attention
        - mask logits
        - validate output

    Quyết định constraint_type nằm ở CSV:

        soft:
            không tạo FSA

        hard:
            tạo FSA, attention-triggered

        protected:
            tạo FSA, attention-triggered
            dùng cho API / JSON / logits / Transformer

    Quan trọng:
        FSA ban đầu ở trạng thái IDLE.
        Chỉ ConstraintActivator mới được gọi fsa.activate()
        khi cross-attention đi vào source span.
    """

    def __init__(
        self,
        tokenizer,
        use_leading_space: bool = True,
    ):
        self.tokenizer = tokenizer
        self.use_leading_space = use_leading_space

    # --------------------------------------------------
    # Normalize
    # --------------------------------------------------

    def _normalize_target_phrase(
        self,
        target_phrase: Any,
    ) -> str:
        return normalize_space(
            target_phrase
        )

    def _normalize_constraint_type(
        self,
        constraint: Constraint,
    ) -> ConstraintType:
        force = parse_bool(
            getattr(
                constraint,
                "force",
                False,
            ),
            default=False,
        )

        protect = parse_bool(
            getattr(
                constraint,
                "protect",
                False,
            ),
            default=False,
        )

        raw_type = getattr(
            constraint,
            "constraint_type",
            None,
        )

        return normalize_constraint_type(
            raw_type,
            force=force,
            protect=protect,
        )

    def _should_build_fsa(
        self,
        constraint: Constraint,
    ) -> bool:
        force = parse_bool(
            getattr(
                constraint,
                "force",
                False,
            ),
            default=False,
        )

        if not force:
            return False

        constraint_type = self._normalize_constraint_type(
            constraint
        )

        return constraint_type in {
            ConstraintType.HARD,
            ConstraintType.PROTECTED,
        }

    # --------------------------------------------------
    # Special tokens
    # --------------------------------------------------

    def _special_token_ids(
        self,
    ) -> set:
        special_ids = set()

        for token_id in [
            getattr(
                self.tokenizer,
                "bos_token_id",
                None,
            ),
            getattr(
                self.tokenizer,
                "eos_token_id",
                None,
            ),
            getattr(
                self.tokenizer,
                "pad_token_id",
                None,
            ),
        ]:
            if token_id is not None:
                special_ids.add(
                    int(
                        token_id
                    )
                )

        return special_ids

    def _unk_token_id(
        self,
    ) -> Optional[int]:
        unk_id = getattr(
            self.tokenizer,
            "unk_token_id",
            None,
        )

        if unk_id is None:
            return None

        return int(
            unk_id
        )

    def _remove_special_ids(
        self,
        token_ids: List[int],
    ) -> List[int]:
        special_ids = self._special_token_ids()

        return [
            int(
                token_id
            )
            for token_id in token_ids
            if int(
                token_id
            )
            not in special_ids
        ]

    def _has_unk(
        self,
        token_ids: List[int],
    ) -> bool:
        unk_id = self._unk_token_id()

        if unk_id is None:
            return False

        return any(
            int(
                token_id
            )
            == unk_id
            for token_id in token_ids
        )

    # --------------------------------------------------
    # Tokenization
    # --------------------------------------------------

    def _candidate_texts(
        self,
        target_phrase: str,
    ) -> List[str]:
        """
        Với SentencePiece/Marian, phrase ở giữa câu thường cần leading space.

        Ví dụ:
            "bộ giải mã"
            " bộ giải mã"

        Ta thử cả hai và chọn candidate hợp lệ đầu tiên.
        """

        target_phrase = target_phrase.strip()

        if not target_phrase:
            return []

        if self.use_leading_space:
            return [
                " " + target_phrase,
                target_phrase,
            ]

        return [
            target_phrase,
            " " + target_phrase,
        ]

    def _encode_raw(
        self,
        text: str,
    ) -> List[int]:
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        return self._remove_special_ids(
            [
                int(
                    token_id
                )
                for token_id in token_ids
            ]
        )

    def _tokens_from_ids(
        self,
        token_ids: List[int],
    ) -> List[str]:
        try:
            return list(
                self.tokenizer.convert_ids_to_tokens(
                    token_ids
                )
            )
        except Exception:
            return []

    def _decode_ids(
        self,
        token_ids: List[int],
    ) -> str:
        try:
            return self.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
        except Exception:
            return ""

    def _encode_candidates(
        self,
        target_phrase: str,
    ) -> List[Dict[str, Any]]:
        candidates = []

        for text in self._candidate_texts(
            target_phrase
        ):
            token_ids = self._encode_raw(
                text
            )

            candidates.append(
                {
                    "text": text,
                    "token_ids": token_ids,
                    "tokens": self._tokens_from_ids(
                        token_ids
                    ),
                    "decoded": self._decode_ids(
                        token_ids
                    ),
                    "has_unk": self._has_unk(
                        token_ids
                    ),
                    "length": len(
                        token_ids
                    ),
                }
            )

        return candidates

    def _select_best_encoding(
        self,
        target_phrase: str,
    ) -> Dict[str, Any]:
        candidates = self._encode_candidates(
            target_phrase
        )

        valid = [
            candidate
            for candidate in candidates
            if candidate[
                "token_ids"
            ]
            and not candidate[
                "has_unk"
            ]
        ]

        if not valid:
            raise ValueError(
                "Không encode được target_phrase hợp lệ cho FSA. "
                f"target_phrase={target_phrase!r}, "
                f"candidates={candidates}"
            )

        # Ưu tiên candidate đầu tiên.
        # Nếu use_leading_space=True thì candidate đầu tiên là " target".
        return valid[
            0
        ]

    def _encode_target_phrase(
        self,
        target_phrase: str,
    ) -> Dict[str, Any]:
        target_phrase = self._normalize_target_phrase(
            target_phrase
        )

        if not target_phrase:
            raise ValueError(
                "target_phrase rỗng, không thể build FSA."
            )

        return self._select_best_encoding(
            target_phrase
        )

    # --------------------------------------------------
    # Build one
    # --------------------------------------------------

    def build_for_constraint(
        self,
        constraint: Constraint,
    ) -> Constraint:
        """
        Build FSA cho một Constraint.

        Với SOFT:
            fsa = None
            force = False

        Với HARD/PROTECTED:
            encode target_phrase
            tạo ConstraintFSA state=IDLE
            force = True
        """

        if constraint is None:
            raise ValueError(
                "constraint is None"
            )

        target_phrase = self._normalize_target_phrase(
            getattr(
                constraint,
                "target_phrase",
                "",
            )
        )

        if not target_phrase:
            raise ValueError(
                "Constraint có target_phrase rỗng: "
                f"{getattr(constraint, 'source_phrase', None)}"
            )

        constraint.target_phrase = target_phrase

        constraint_type = self._normalize_constraint_type(
            constraint
        )

        constraint.constraint_type = constraint_type

        # --------------------------------------------------
        # Không force hoặc SOFT: không tạo FSA.
        # --------------------------------------------------

        if not self._should_build_fsa(constraint):
            constraint.force = False
            constraint.fsa = None
            constraint.target_token_ids = []
            constraint.target_tokens = []

            constraint.meta[
                "fsa_built"
            ] = False

            constraint.meta[
                "fsa_reason"
            ] = "not_forced_or_soft_constraint"

            return constraint

        # --------------------------------------------------
        # HARD / PROTECTED: tạo FSA.
        # --------------------------------------------------

        encoding = self._encode_target_phrase(
            target_phrase
        )

        target_token_ids = [
            int(
                token_id
            )
            for token_id in encoding[
                "token_ids"
            ]
        ]

        if not target_token_ids:
            raise ValueError(
                f"target_phrase không có token IDs: {target_phrase}"
            )

        constraint.force = True
        constraint.target_token_ids = target_token_ids
        constraint.target_tokens = list(
            encoding[
                "tokens"
            ]
        )

        # FSA mới luôn IDLE.
        # AttentionActivator sẽ gọi activate() khi attention đúng source span.
        constraint.fsa = ConstraintFSA(
            target_phrase=target_phrase,
            target_token_ids=target_token_ids,
            meta={
                "constraint_id": constraint.id,
                "constraint_type": constraint_type.value
                if hasattr(
                    constraint_type,
                    "value",
                )
                else str(
                    constraint_type
                ),
                "source_phrase": constraint.source_phrase,
                "target_phrase": target_phrase,
                "decoded_target": encoding[
                    "decoded"
                ],
            },
        )

        constraint.meta[
            "fsa_built"
        ] = True

        constraint.meta[
            "encoded_target"
        ] = {
            "token_ids": list(
                target_token_ids
            ),
            "tokens": list(
                encoding[
                    "tokens"
                ]
            ),
            "decoded": encoding[
                "decoded"
            ],
        }

        return constraint

    # --------------------------------------------------
    # Build all
    # --------------------------------------------------

    def build_all(
        self,
        constraints: List[Constraint],
    ) -> List[Constraint]:
        if not constraints:
            return []

        return [
            self.build_for_constraint(
                constraint
            )
            for constraint in constraints
        ]

    # --------------------------------------------------
    # Compact inspection
    # --------------------------------------------------

    def summary(
        self,
        constraints: List[Constraint],
    ) -> List[Dict[str, Any]]:
        """
        Debug ngắn, dùng khi cần kiểm tra nhanh.
        Không dùng trace dài trong UI.
        """

        result = []

        for constraint in constraints:
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

            result.append(
                {
                    "source": getattr(
                        constraint,
                        "source_phrase",
                        "",
                    ),
                    "target": getattr(
                        constraint,
                        "target_phrase",
                        "",
                    ),
                    "type": constraint_type,
                    "force": getattr(
                        constraint,
                        "force",
                        False,
                    ),
                    "has_fsa": getattr(
                        constraint,
                        "fsa",
                        None,
                    )
                    is not None,
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
                    "token_span": getattr(
                        constraint,
                        "token_span",
                        None,
                    ),
                }
            )

        return result