from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from terminology.constraint import (
    Constraint,
    ConstraintType,
)


class LogitsMasker:
    """
    Hard-mask logits theo ACTIVE constraint.

    Vai trò duy nhất:

        ACTIVE Constraint
        -> Constraint.fsa.allowed_token_ids()
        -> mask logits, chỉ giữ token hợp lệ

    File này KHÔNG:
        - đọc attention
        - activate constraint
        - build FSA
        - dò target phrase trong generated_text
        - mark DONE bằng text
        - sửa output sau dịch

    Flow đúng:

        AttentionMonitor
            -> AttentionInfo

        ConstraintActivator
            -> constraint.activate()
            -> fsa.state = FORCING

        LogitsMasker
            -> allowed_token_ids = [next_token_id]
            -> mask logits

        BeamSearch chọn token

        FSA.step(token_id)

        Nếu fsa DONE:
            constraint.mark_done()
    """

    def __init__(
        self,
        tokenizer=None,
        strict: bool = True,
        mask_eos_until_forced_done: bool = True,
    ):
        self.tokenizer = tokenizer
        self.strict = bool(
            strict
        )
        self.mask_eos_until_forced_done = bool(
            mask_eos_until_forced_done
        )

        self.last_debug: Dict[str, Any] = {}

    # --------------------------------------------------
    # Debug
    # --------------------------------------------------

    def _reset_debug(
        self,
    ) -> None:
        self.last_debug = {
            "mask_applied": False,
            "reason": None,
            "active_constraint_id": None,
            "active_source_phrase": None,
            "active_target_phrase": None,
            "constraint_type": None,
            "allowed_token_ids": [],
            "allowed_tokens": [],
            "vocab_size": None,
            "eos_blocked": False,
            "eos_token_id": None,
        }

    def debug_info(
        self,
    ) -> Dict[str, Any]:
        return dict(
            self.last_debug
        )

    # --------------------------------------------------
    # Token helpers
    # --------------------------------------------------

    def _vocab_size(
        self,
        logits: torch.Tensor,
    ) -> int:
        return int(
            logits.size(
                -1
            )
        )

    def _token_from_id(
        self,
        token_id: int,
    ) -> Optional[str]:
        if self.tokenizer is None:
            return None

        try:
            return self.tokenizer.convert_ids_to_tokens(
                int(
                    token_id
                )
            )
        except Exception:
            return None

    def _tokens_from_ids(
        self,
        token_ids: Sequence[int],
    ) -> List[Optional[str]]:
        return [
            self._token_from_id(
                int(
                    token_id
                )
            )
            for token_id in token_ids
        ]

    def _get_eos_token_id(
        self,
        eos_token_id: Optional[int] = None,
    ) -> Optional[int]:
        if eos_token_id is not None:
            return int(
                eos_token_id
            )

        if self.tokenizer is None:
            return None

        value = getattr(
            self.tokenizer,
            "eos_token_id",
            None,
        )

        if value is None:
            return None

        return int(
            value
        )

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
            )

        return str(
            value
        )

    def _is_forceable(
        self,
        constraint: Optional[Constraint],
    ) -> bool:
        if constraint is None:
            return False

        constraint_type = getattr(
            constraint,
            "constraint_type",
            ConstraintType.SOFT,
        )

        if constraint_type == ConstraintType.SOFT:
            return False

        if self._constraint_type_value(
            constraint
        ).lower() == "soft":
            return False

        if not bool(
            getattr(
                constraint,
                "force",
                False,
            )
        ):
            return False

        if getattr(
            constraint,
            "fsa",
            None,
        ) is None:
            return False

        return True

    def _is_done(
        self,
        constraint: Constraint,
    ) -> bool:
        if hasattr(
            constraint,
            "is_done",
        ):
            try:
                return bool(
                    constraint.is_done()
                )
            except Exception:
                pass

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
        ).upper() == "DONE"

    def _is_active(
        self,
        constraint: Constraint,
    ) -> bool:
        if hasattr(
            constraint,
            "is_active",
        ):
            try:
                return bool(
                    constraint.is_active()
                )
            except Exception:
                pass

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
        ).upper() == "ACTIVE"

    def _find_active_constraint(
        self,
        constraints: Optional[Iterable[Constraint]],
    ) -> Optional[Constraint]:
        if constraints is None:
            return None

        for constraint in constraints:
            if not self._is_forceable(
                constraint
            ):
                continue

            if self._is_active(
                constraint
            ):
                return constraint

        return None

    def _all_forced_constraints_done(
        self,
        constraints: Optional[Iterable[Constraint]],
    ) -> bool:
        if constraints is None:
            return True

        forced_constraints = [
            constraint
            for constraint in constraints
            if self._is_forceable(
                constraint
            )
        ]

        if not forced_constraints:
            return True

        return all(
            self._is_done(
                constraint
            )
            for constraint in forced_constraints
        )

    def _allowed_token_ids_from_constraint(
        self,
        constraint: Optional[Constraint],
    ) -> List[int]:
        if constraint is None:
            return []

        if not self._is_forceable(
            constraint
        ):
            return []

        if not self._is_active(
            constraint
        ):
            return []

        if hasattr(
            constraint,
            "allowed_token_ids",
        ):
            try:
                return [
                    int(
                        token_id
                    )
                    for token_id in constraint.allowed_token_ids()
                ]
            except Exception:
                return []

        fsa = getattr(
            constraint,
            "fsa",
            None,
        )

        if fsa is None:
            return []

        if hasattr(
            fsa,
            "allowed_token_ids",
        ):
            try:
                return [
                    int(
                        token_id
                    )
                    for token_id in fsa.allowed_token_ids()
                ]
            except Exception:
                return []

        if hasattr(
            fsa,
            "next_token_id",
        ):
            try:
                token_id = fsa.next_token_id()

                if token_id is None:
                    return []

                return [
                    int(
                        token_id
                    )
                ]
            except Exception:
                return []

        return []

    # --------------------------------------------------
    # Mask helpers
    # --------------------------------------------------

    def _filter_valid_token_ids(
        self,
        token_ids: Sequence[int],
        vocab_size: int,
    ) -> List[int]:
        result = []

        seen = set()

        for token_id in token_ids:
            token_id = int(
                token_id
            )

            if token_id < 0 or token_id >= vocab_size:
                continue

            if token_id in seen:
                continue

            seen.add(
                token_id
            )

            result.append(
                token_id
            )

        return result

    def _mask_to_allowed(
        self,
        logits: torch.Tensor,
        allowed_token_ids: Sequence[int],
    ) -> torch.Tensor:
        vocab_size = self._vocab_size(
            logits
        )

        allowed_token_ids = self._filter_valid_token_ids(
            allowed_token_ids,
            vocab_size=vocab_size,
        )

        if not allowed_token_ids:
            message = (
                "ACTIVE constraint không có allowed token hợp lệ. "
                f"vocab_size={vocab_size}"
            )

            self.last_debug[
                "reason"
            ] = message

            if self.strict:
                raise ValueError(
                    message
                )

            return logits

        masked_logits = torch.full_like(
            logits,
            float(
                "-inf"
            ),
        )

        ids_tensor = torch.tensor(
            allowed_token_ids,
            device=logits.device,
            dtype=torch.long,
        )

        masked_logits.index_copy_(
            dim=-1,
            index=ids_tensor,
            source=logits.index_select(
                dim=-1,
                index=ids_tensor,
            ),
        )

        self.last_debug[
            "mask_applied"
        ] = True

        self.last_debug[
            "reason"
        ] = "active_fsa_hard_mask"

        self.last_debug[
            "allowed_token_ids"
        ] = list(
            allowed_token_ids
        )

        self.last_debug[
            "allowed_tokens"
        ] = self._tokens_from_ids(
            allowed_token_ids
        )

        return masked_logits

    def _block_token(
        self,
        logits: torch.Tensor,
        token_id: Optional[int],
    ) -> torch.Tensor:
        if token_id is None:
            return logits

        vocab_size = self._vocab_size(
            logits
        )

        token_id = int(
            token_id
        )

        if token_id < 0 or token_id >= vocab_size:
            return logits

        masked_logits = logits.clone()

        masked_logits[
            ...,
            token_id,
        ] = float(
            "-inf"
        )

        return masked_logits

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def apply(
        self,
        logits: torch.Tensor,
        active_constraint: Optional[Constraint] = None,
        constraints: Optional[List[Constraint]] = None,
        eos_token_id: Optional[int] = None,
        generated_text: Optional[str] = None,
        recent_done_constraints=None,
    ) -> torch.Tensor:
        """
        Mask logits ở một decoding step.

        Args:
            logits:
                Tensor shape [..., vocab_size]

            active_constraint:
                Constraint đang ACTIVE.
                Nếu None, masker sẽ tự tìm ACTIVE constraint trong constraints.

            constraints:
                Toàn bộ constraints của hypothesis.
                Dùng để block EOS nếu còn forced constraint chưa DONE.

            eos_token_id:
                Optional. Nếu không truyền, lấy tokenizer.eos_token_id.

            generated_text:
                Giữ lại để tương thích với code cũ.
                Bản mới KHÔNG dùng generated_text để mark DONE.

            recent_done_constraints:
                Giữ lại để tương thích với code cũ.
                Bản mới KHÔNG boundary-guard bằng text.

        Return:
            logits đã mask.
        """

        self._reset_debug()

        if logits is None:
            self.last_debug[
                "reason"
            ] = "logits_none"

            return logits

        self.last_debug[
            "vocab_size"
        ] = self._vocab_size(
            logits
        )

        # Nếu không truyền active_constraint, tự tìm trong constraints.
        if active_constraint is None:
            active_constraint = self._find_active_constraint(
                constraints
            )

        # --------------------------------------------------
        # 1. Nếu có ACTIVE FSA, hard-mask theo allowed token.
        # --------------------------------------------------

        if active_constraint is not None:
            self.last_debug[
                "active_constraint_id"
            ] = getattr(
                active_constraint,
                "id",
                None,
            )

            self.last_debug[
                "active_source_phrase"
            ] = getattr(
                active_constraint,
                "source_phrase",
                None,
            )

            self.last_debug[
                "active_target_phrase"
            ] = getattr(
                active_constraint,
                "target_phrase",
                None,
            )

            self.last_debug[
                "constraint_type"
            ] = self._constraint_type_value(
                active_constraint
            )

            allowed_token_ids = self._allowed_token_ids_from_constraint(
                active_constraint
            )

            if allowed_token_ids:
                return self._mask_to_allowed(
                    logits=logits,
                    allowed_token_ids=allowed_token_ids,
                )

            # Có active constraint nhưng FSA không còn token.
            # Thường xảy ra nếu FSA đã DONE nhưng constraint chưa mark_done.
            self.last_debug[
                "reason"
            ] = "active_constraint_without_allowed_tokens"

        # --------------------------------------------------
        # 2. Nếu chưa active constraint nào, có thể block EOS
        #    cho đến khi tất cả forced constraints DONE.
        # --------------------------------------------------

        eos_id = self._get_eos_token_id(
            eos_token_id
        )

        self.last_debug[
            "eos_token_id"
        ] = eos_id

        if (
            self.mask_eos_until_forced_done
            and eos_id is not None
            and not self._all_forced_constraints_done(
                constraints
            )
        ):
            logits = self._block_token(
                logits=logits,
                token_id=eos_id,
            )

            self.last_debug[
                "eos_blocked"
            ] = True

            if self.last_debug.get(
                "reason"
            ) is None:
                self.last_debug[
                    "reason"
                ] = "eos_blocked_until_forced_constraints_done"

            return logits

        if self.last_debug.get(
            "reason"
        ) is None:
            self.last_debug[
                "reason"
            ] = "no_mask_needed"

        return logits

    def mask(
        self,
        logits: torch.Tensor,
        active_constraint: Optional[Constraint] = None,
        constraints: Optional[List[Constraint]] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Alias ngắn cho apply().
        """

        return self.apply(
            logits=logits,
            active_constraint=active_constraint,
            constraints=constraints,
            eos_token_id=eos_token_id,
        )