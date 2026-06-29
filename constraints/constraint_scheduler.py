from typing import Any, List, Optional


class ConstraintScheduler:
    """
    Điều phối constraint trong lúc decoding.

    Bản này dùng cho cơ chế hiện tại:

        Attention / source-order activation
        + STRICT_HARD / RELAXED_HARD FSA
        + không ép fallback bừa bãi
        + mặc định không block EOS

    Vai trò chính:
        1. Không tự activate constraint nếu enable_fallback=False.
        2. Không block EOS nếu block_eos_until_done=False.
        3. Nếu sau này bật fallback/block EOS thì chỉ áp dụng cho HARD constraint:
            force=True
            fsa != None

    Lưu ý:
        ConstraintActivator mới đã xử lý:
            - attention_argmax
            - attention_topk
            - source_order_near_attention
            - source_order_progress_activation

        Vì vậy Scheduler không nên tự ép thêm constraint nữa,
        nếu không sẽ dễ làm câu bị dư thuật ngữ.
    """

    def __init__(
        self,
        fallback_after_step: int = 999999,
        block_eos_until_done: bool = False,
        enable_fallback: bool = False,
        block_eos_only_when_active: bool = True,
    ):
        self.fallback_after_step = fallback_after_step
        self.block_eos_until_done = block_eos_until_done
        self.enable_fallback = enable_fallback
        self.block_eos_only_when_active = block_eos_only_when_active

        self.last_debug = None

    # --------------------------------------------------
    # Debug
    # --------------------------------------------------

    def _reset_debug(
        self,
    ):
        self.last_debug = {
            "enable_fallback": self.enable_fallback,
            "fallback_after_step": self.fallback_after_step,
            "block_eos_until_done": self.block_eos_until_done,
            "block_eos_only_when_active": self.block_eos_only_when_active,

            "eos_blocked": False,
            "eos_block_reason": None,
            "fallback_selected": False,
            "fallback_reason": None,

            "hard_total": 0,
            "hard_done": 0,
            "hard_unfinished": 0,
            "has_active_hard_constraint": False,
            "active_constraint": None,
            "fallback_constraint": None,
        }

    def debug_info(
        self,
    ):
        return self.last_debug

    def _constraint_debug(
        self,
        constraint,
    ):
        if constraint is None:
            return None

        fsa = getattr(
            constraint,
            "fsa",
            None,
        )

        return {
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
            "state": getattr(
                constraint,
                "state",
                None,
            ),
            "force": getattr(
                constraint,
                "force",
                None,
            ),
            "constraint_type": getattr(
                constraint,
                "constraint_type",
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
            "has_fsa": fsa is not None,
            "fsa_mode": getattr(
                fsa,
                "mode",
                None,
            )
            if fsa is not None
            else None,
            "fsa_phase": getattr(
                fsa,
                "phase",
                None,
            )
            if fsa is not None
            else None,
            "fsa_position": getattr(
                fsa,
                "position",
                None,
            )
            if fsa is not None
            else None,
            "fsa_done": getattr(
                fsa,
                "is_done",
                None,
            )
            if fsa is not None
            else None,
        }

    # --------------------------------------------------
    # Constraint checks
    # --------------------------------------------------

    def _has_fsa(
        self,
        constraint,
    ) -> bool:
        if constraint is None:
            return False

        return getattr(
            constraint,
            "fsa",
            None,
        ) is not None

    def _is_forced(
        self,
        constraint,
    ) -> bool:
        if constraint is None:
            return False

        return getattr(
            constraint,
            "force",
            False,
        ) is True

    def _is_hard_constraint(
        self,
        constraint,
    ) -> bool:
        """
        HARD constraint:
            force=True
            và có FSA

        Bao gồm:
            STRICT_HARD
            RELAXED_HARD
        """

        return self._is_forced(
            constraint
        ) and self._has_fsa(
            constraint
        )

    def _fsa_is_done(
        self,
        constraint,
    ) -> bool:
        if constraint is None:
            return False

        fsa = getattr(
            constraint,
            "fsa",
            None,
        )

        if fsa is None:
            return False

        return bool(
            getattr(
                fsa,
                "is_done",
                False,
            )
        )

    def _is_done(
        self,
        constraint,
        constraint_manager=None,
    ) -> bool:
        if constraint is None:
            return False

        if constraint_manager is not None:
            try:
                return bool(
                    constraint_manager.is_done(
                        constraint
                    )
                )
            except Exception:
                pass

        if getattr(
            constraint,
            "state",
            None,
        ) == "DONE":
            return True

        if self._fsa_is_done(
            constraint
        ):
            return True

        return False

    def _is_active(
        self,
        constraint,
    ) -> bool:
        if constraint is None:
            return False

        return getattr(
            constraint,
            "state",
            None,
        ) == "ACTIVE"

    def _hard_constraints(
        self,
        constraints,
    ):
        return [
            constraint
            for constraint in constraints or []
            if self._is_hard_constraint(
                constraint
            )
        ]

    def _unfinished_hard_constraints(
        self,
        constraints,
        constraint_manager=None,
    ):
        return [
            constraint
            for constraint in self._hard_constraints(
                constraints
            )
            if not self._is_done(
                constraint,
                constraint_manager=constraint_manager,
            )
        ]

    def _active_hard_constraint(
        self,
        constraints,
    ):
        for constraint in constraints or []:
            if not self._is_hard_constraint(
                constraint
            ):
                continue

            if self._is_active(
                constraint
            ):
                return constraint

        return None

    def _update_constraint_counts(
        self,
        constraints,
        constraint_manager=None,
    ):
        if self.last_debug is None:
            return

        hard_constraints = self._hard_constraints(
            constraints
        )

        unfinished = self._unfinished_hard_constraints(
            constraints,
            constraint_manager=constraint_manager,
        )

        hard_done = len(
            hard_constraints
        ) - len(
            unfinished
        )

        active = self._active_hard_constraint(
            constraints
        )

        self.last_debug["hard_total"] = len(
            hard_constraints
        )
        self.last_debug["hard_done"] = hard_done
        self.last_debug["hard_unfinished"] = len(
            unfinished
        )
        self.last_debug["has_active_hard_constraint"] = active is not None
        self.last_debug["active_constraint"] = self._constraint_debug(
            active
        )

    # --------------------------------------------------
    # Public status
    # --------------------------------------------------

    def has_unfinished_constraints(
        self,
        constraints: List[Any],
        constraint_manager,
    ) -> bool:
        """
        Kiểm tra còn HARD constraint nào chưa DONE không.

        SOFT constraint không tính.
        """

        if not constraints:
            return False

        unfinished = self._unfinished_hard_constraints(
            constraints,
            constraint_manager=constraint_manager,
        )

        return len(
            unfinished
        ) > 0

    # --------------------------------------------------
    # EOS blocking
    # --------------------------------------------------

    def block_eos(
        self,
        logits,
        constraints: List[Any],
        eos_token_id: Optional[int],
        constraint_manager,
    ):
        """
        Mặc định KHÔNG chặn EOS.

        Nếu block_eos_until_done=True:
            chỉ chặn EOS khi còn HARD constraint chưa DONE.

        Nếu block_eos_only_when_active=True:
            chỉ chặn EOS khi đang có ACTIVE HARD constraint.
            Cách này an toàn hơn, tránh model sinh rác nếu attention
            không bao giờ kích hoạt constraint còn lại.

        Return:
            logits, eos_blocked
        """

        self._reset_debug()
        self._update_constraint_counts(
            constraints,
            constraint_manager=constraint_manager,
        )

        if not self.block_eos_until_done:
            self.last_debug["eos_block_reason"] = "block_eos_disabled"
            return logits, False

        if logits is None:
            self.last_debug["eos_block_reason"] = "logits_none"
            return logits, False

        if eos_token_id is None:
            self.last_debug["eos_block_reason"] = "eos_token_id_none"
            return logits, False

        if not self.has_unfinished_constraints(
            constraints,
            constraint_manager,
        ):
            self.last_debug["eos_block_reason"] = "no_unfinished_hard_constraints"
            return logits, False

        active_hard = self._active_hard_constraint(
            constraints
        )

        if self.block_eos_only_when_active and active_hard is None:
            self.last_debug["eos_block_reason"] = "no_active_hard_constraint"
            return logits, False

        new_logits = logits.clone()

        new_logits[
            :,
            int(
                eos_token_id
            ),
        ] = float(
            "-inf"
        )

        self.last_debug["eos_blocked"] = True
        self.last_debug["eos_block_reason"] = "blocked_until_active_or_unfinished_hard_done"

        return new_logits, True

    # --------------------------------------------------
    # Fallback activation
    # --------------------------------------------------

    def _pending_hard_constraints(
        self,
        constraints,
        constraint_manager,
    ):
        """
        Lấy HARD constraints chưa DONE, ưu tiên NOT_STARTED.
        """

        pending = []

        for constraint in constraints or []:
            if not self._is_hard_constraint(
                constraint
            ):
                continue

            if self._is_done(
                constraint,
                constraint_manager=constraint_manager,
            ):
                continue

            if getattr(
                constraint,
                "state",
                None,
            ) == "ACTIVE":
                continue

            pending.append(
                constraint
            )

        return sorted(
            pending,
            key=lambda c: (
                getattr(
                    c,
                    "word_span",
                    [
                        999999,
                        999999,
                    ],
                )[0],
                getattr(
                    c,
                    "word_span",
                    [
                        999999,
                        999999,
                    ],
                )[1],
            ),
        )

    def maybe_fallback_activate(
        self,
        step: int,
        constraints: List[Any],
        constraint_manager,
    ):
        """
        Mặc định KHÔNG fallback.

        Nếu enable_fallback=False:
            return None.

        Nếu bật fallback:
            chỉ fallback HARD constraint,
            không fallback SOFT constraint.
        """

        self._reset_debug()
        self._update_constraint_counts(
            constraints,
            constraint_manager=constraint_manager,
        )

        if not self.enable_fallback:
            self.last_debug["fallback_reason"] = "fallback_disabled"
            return None

        if not constraints:
            self.last_debug["fallback_reason"] = "no_constraints"
            return None

        if constraint_manager.has_active_constraint(
            constraints
        ):
            self.last_debug["fallback_reason"] = "has_active_constraint"
            return None

        if step < self.fallback_after_step:
            self.last_debug["fallback_reason"] = "fallback_step_not_reached"
            return None

        pending = self._pending_hard_constraints(
            constraints,
            constraint_manager=constraint_manager,
        )

        if not pending:
            self.last_debug["fallback_reason"] = "no_pending_hard_constraints"
            return None

        constraint = pending[
            0
        ]

        activated = constraint_manager.activate(
            constraint
        )

        self.last_debug["fallback_selected"] = activated is not None
        self.last_debug["fallback_constraint"] = self._constraint_debug(
            activated
        )
        self.last_debug["fallback_reason"] = "fallback_activated_hard_constraint"

        return activated

    # --------------------------------------------------
    # Mode helper
    # --------------------------------------------------

    def should_use_attention_only(
        self,
    ) -> bool:
        """
        Dùng cho debug/logging.

        True nghĩa là Scheduler không tự fallback và không block EOS.
        """

        return (
            not self.enable_fallback
            and not self.block_eos_until_done
        )