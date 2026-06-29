import re


class ConstraintManager:
    """
    Quản lý trạng thái terminology constraints.

    Trạng thái FSA/HARD:
        NOT_STARTED
        ACTIVE
        DONE

    Trạng thái SOFT:
        SOFT_FOUND
        SOFT_MISSING

    Quy tắc mới:

    1. SOFT constraint:
        - force=False hoặc fsa=None
        - không activate
        - không mask logits
        - không chuyển sang DONE bằng FSA
        - chỉ kiểm tra target_phrase có xuất hiện trong output không

    2. RELAXED_HARD constraint:
        - force=True
        - fsa.mode="relaxed"
        - được activate
        - FSA cho phép token tự do trước/sau constraint

    3. STRICT_HARD constraint:
        - force=True
        - fsa.mode="strict"
        - chỉ dùng rất ít
        - không dùng cho acronym/thường ngữ như API, JSON, logits

    4. Không dùng protect=1 để suy ra strict_hard.
       protect chỉ là giữ nguyên mặt chữ.

    5. HARD constraint chỉ DONE khi:
        - FSA DONE
        - hoặc được sync an toàn với chính selected/active constraint
    """

    NOT_STARTED = "NOT_STARTED"
    ACTIVE = "ACTIVE"
    DONE = "DONE"

    SOFT_FOUND = "SOFT_FOUND"
    SOFT_MISSING = "SOFT_MISSING"

    # --------------------------------------------------
    # Text utility
    # --------------------------------------------------

    def _normalize_text(
        self,
        text: str,
    ) -> str:
        if text is None:
            return ""

        text = str(
            text
        ).lower()

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

    def _normalize_for_match(
        self,
        text: str,
    ) -> str:
        """
        Normalize nhẹ để match target phrase trong output.

        Không xóa chữ cái.
        Không đổi nghĩa.
        Chỉ xử lý khoảng trắng và dấu câu biên.
        """

        text = self._normalize_text(
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    def target_exists_in_text(
        self,
        target_phrase: str,
        text: str,
    ) -> bool:
        """
        Kiểm tra target_phrase có trong text không.

        Dùng cho:
            - SOFT validation
            - lexical sync có kiểm soát

        Không tự động mark mọi HARD constraint là DONE.
        """

        target = self._normalize_for_match(
            target_phrase
        )

        output = self._normalize_for_match(
            text
        )

        if not target or not output:
            return False

        return target in output

    def target_ends_output(
        self,
        target_phrase: str,
        text: str,
    ) -> bool:
        """
        Kiểm tra output hiện tại có kết thúc bằng target_phrase không.

        Dùng để tránh ép lại một constraint vừa được model sinh ra.
        """

        target = self._normalize_for_match(
            target_phrase
        )

        output = self._normalize_for_match(
            text
        )

        if not target or not output:
            return False

        return output.endswith(
            target
        )

    # --------------------------------------------------
    # Constraint type checks
    # --------------------------------------------------

    def has_fsa(
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

    def is_forced(
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

    def constraint_type(
        self,
        constraint,
    ):
        if constraint is None:
            return None

        return getattr(
            constraint,
            "constraint_type",
            None,
        )

    def is_soft(
        self,
        constraint,
    ) -> bool:
        if constraint is None:
            return False

        ctype = str(
            self.constraint_type(
                constraint
            )
            or ""
        ).lower()

        if ctype == "soft":
            return True

        if not self.is_forced(
            constraint
        ):
            return True

        if not self.has_fsa(
            constraint
        ):
            return True

        return False

    def is_hard(
        self,
        constraint,
    ) -> bool:
        if constraint is None:
            return False

        if self.is_soft(
            constraint
        ):
            return False

        return self.is_forced(
            constraint
        ) and self.has_fsa(
            constraint
        )

    def fsa_mode(
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

        if fsa is None:
            return None

        return getattr(
            fsa,
            "mode",
            None,
        )

    def is_relaxed_hard(
        self,
        constraint,
    ) -> bool:
        return (
            self.is_hard(
                constraint
            )
            and self.fsa_mode(
                constraint
            )
            == "relaxed"
        )

    def is_strict_hard(
        self,
        constraint,
    ) -> bool:
        return (
            self.is_hard(
                constraint
            )
            and self.fsa_mode(
                constraint
            )
            == "strict"
        )

    # --------------------------------------------------
    # FSA checks
    # --------------------------------------------------

    def fsa_is_done(
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

    def fsa_should_force_token(
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

        if hasattr(
            fsa,
            "should_force_token",
        ):
            try:
                return bool(
                    fsa.should_force_token()
                )
            except Exception:
                return False

        if hasattr(
            fsa,
            "next_token_id",
        ):
            try:
                return fsa.next_token_id() is not None
            except Exception:
                return False

        return False

    def fsa_debug(
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

        if fsa is None:
            return None

        if hasattr(
            fsa,
            "to_dict",
        ):
            try:
                return fsa.to_dict()
            except Exception:
                pass

        return {
            "target_phrase": getattr(
                fsa,
                "target_phrase",
                None,
            ),
            "target_token_ids": getattr(
                fsa,
                "target_token_ids",
                None,
            ),
            "position": getattr(
                fsa,
                "position",
                None,
            ),
            "target_length": len(
                getattr(
                    fsa,
                    "target_token_ids",
                    [],
                )
                or []
            ),
            "mode": getattr(
                fsa,
                "mode",
                None,
            ),
            "phase": getattr(
                fsa,
                "phase",
                None,
            ),
            "is_done": getattr(
                fsa,
                "is_done",
                None,
            ),
            "max_extra_before": getattr(
                fsa,
                "max_extra_before",
                None,
            ),
            "max_extra_after": getattr(
                fsa,
                "max_extra_after",
                None,
            ),
            "extra_before_used": getattr(
                fsa,
                "extra_before_used",
                None,
            ),
            "extra_after_used": getattr(
                fsa,
                "extra_after_used",
                None,
            ),
        }

    # --------------------------------------------------
    # State checks
    # --------------------------------------------------

    def is_done(
        self,
        constraint,
    ) -> bool:
        if constraint is None:
            return False

        if self.is_soft(
            constraint
        ):
            return False

        if getattr(
            constraint,
            "state",
            None,
        ) == self.DONE:
            return True

        if self.fsa_is_done(
            constraint
        ):
            return True

        return False

    def is_active(
        self,
        constraint,
    ) -> bool:
        if constraint is None:
            return False

        return getattr(
            constraint,
            "state",
            None,
        ) == self.ACTIVE

    def is_not_started(
        self,
        constraint,
    ) -> bool:
        if constraint is None:
            return False

        return getattr(
            constraint,
            "state",
            None,
        ) == self.NOT_STARTED

    def can_activate(
        self,
        constraint,
    ) -> bool:
        """
        Chỉ activate HARD constraint.

        Không activate SOFT constraint.
        Không activate constraint đã DONE.
        """

        if constraint is None:
            return False

        if not self.is_hard(
            constraint
        ):
            return False

        if self.is_done(
            constraint
        ):
            return False

        return True

    # --------------------------------------------------
    # SOFT status
    # --------------------------------------------------

    def soft_satisfied(
        self,
        constraint,
        generated_text: str,
    ) -> bool:
        if constraint is None:
            return False

        if not self.is_soft(
            constraint
        ):
            return False

        return self.target_exists_in_text(
            getattr(
                constraint,
                "target_phrase",
                "",
            ),
            generated_text,
        )

    def soft_status(
        self,
        constraint,
        generated_text: str,
    ) -> str:
        if self.soft_satisfied(
            constraint,
            generated_text,
        ):
            return self.SOFT_FOUND

        return self.SOFT_MISSING

    def display_status(
        self,
        constraint,
        generated_text: str = "",
    ) -> str:
        """
        Trạng thái dùng cho UI.

        HARD:
            NOT_STARTED / ACTIVE / DONE

        SOFT:
            SOFT_FOUND / SOFT_MISSING
        """

        if constraint is None:
            return "NONE"

        if self.is_soft(
            constraint
        ):
            return self.soft_status(
                constraint,
                generated_text,
            )

        if self.is_done(
            constraint
        ):
            return self.DONE

        if self.is_active(
            constraint
        ):
            return self.ACTIVE

        return self.NOT_STARTED

    # --------------------------------------------------
    # Query constraints
    # --------------------------------------------------

    def get_active_constraint(
        self,
        constraints,
    ):
        """
        Lấy constraint đang ACTIVE.

        Nếu ACTIVE nhưng FSA đã DONE:
            mark DONE rồi bỏ qua.
        """

        if not constraints:
            return None

        for constraint in constraints:
            if not self.is_active(
                constraint
            ):
                continue

            if self.fsa_is_done(
                constraint
            ):
                self.mark_done(
                    constraint
                )
                continue

            return constraint

        return None

    def get_not_started(
        self,
        constraints,
    ):
        if not constraints:
            return []

        return [
            constraint
            for constraint in constraints
            if self.is_not_started(
                constraint
            )
        ]

    def get_pending_hard(
        self,
        constraints,
    ):
        if not constraints:
            return []

        return [
            constraint
            for constraint in constraints
            if self.is_hard(
                constraint
            )
            and not self.is_done(
                constraint
            )
        ]

    def get_done(
        self,
        constraints,
    ):
        if not constraints:
            return []

        return [
            constraint
            for constraint in constraints
            if self.is_done(
                constraint
            )
        ]

    def get_soft(
        self,
        constraints,
    ):
        if not constraints:
            return []

        return [
            constraint
            for constraint in constraints
            if self.is_soft(
                constraint
            )
        ]

    def has_active_constraint(
        self,
        constraints,
    ) -> bool:
        if not constraints:
            return False

        return any(
            self.is_active(
                constraint
            )
            for constraint in constraints
        )

    def all_done(
        self,
        constraints,
    ) -> bool:
        """
        Kiểm tra tất cả HARD constraints đã DONE chưa.

        SOFT constraints không tính vào all_done.
        """

        hard_constraints = [
            constraint
            for constraint in constraints or []
            if self.is_hard(
                constraint
            )
        ]

        if not hard_constraints:
            return True

        return all(
            self.is_done(
                constraint
            )
            for constraint in hard_constraints
        )

    def done_spans(
        self,
        constraints,
    ):
        """
        Trả về source token_span đã DONE.

        Chỉ HARD DONE mới nên dùng để mask source span.
        SOFT không mask.
        """

        if not constraints:
            return []

        spans = []

        for constraint in constraints:
            if not self.is_hard(
                constraint
            ):
                continue

            if not self.is_done(
                constraint
            ):
                continue

            token_span = getattr(
                constraint,
                "token_span",
                None,
            )

            if token_span is None:
                continue

            spans.append(
                token_span
            )

        return spans

    # --------------------------------------------------
    # State transitions
    # --------------------------------------------------

    def activate(
        self,
        constraint,
    ):
        """
        Chuyển HARD constraint sang ACTIVE.

        SOFT constraint không được activate.
        Constraint DONE không được activate lại.
        """

        if not self.can_activate(
            constraint
        ):
            return None

        if self.is_not_started(
            constraint
        ):
            constraint.state = self.ACTIVE

        return constraint

    def activate_by_attention(
        self,
        constraint,
    ):
        """
        Giữ tên hàm để không vỡ code cũ.
        """

        return self.activate(
            constraint
        )

    def update_by_attention(
        self,
        active_constraint,
    ):
        """
        Kích hoạt constraint được chọn trước argmax.
        """

        if active_constraint is None:
            return None

        return self.activate_by_attention(
            active_constraint
        )

    def mark_done(
        self,
        constraint,
    ):
        """
        Đánh dấu HARD constraint DONE.

        Không dùng cho SOFT.
        """

        if constraint is None:
            return None

        if self.is_soft(
            constraint
        ):
            return constraint

        constraint.state = self.DONE

        fsa = getattr(
            constraint,
            "fsa",
            None,
        )

        if fsa is not None and hasattr(
            fsa,
            "force_done",
        ):
            try:
                fsa.force_done()
            except Exception:
                pass

        return constraint

    def mark_done_if_fsa_done(
        self,
        constraint,
    ):
        """
        Mark DONE chỉ khi FSA thật sự hoàn thành.
        """

        if constraint is None:
            return None

        if not self.has_fsa(
            constraint
        ):
            return constraint

        if self.fsa_is_done(
            constraint
        ):
            return self.mark_done(
                constraint
            )

        return constraint

    def mark_done_if_target_already_generated(
        self,
        constraint,
        generated_text: str,
        require_output_endswith_target: bool = False,
    ):
        """
        Lexical sync an toàn cho HARD constraint.

        Dùng trước khi ép FSA.

        Mục tiêu:
            Nếu selected/active constraint đã có trong output,
            không ép lặp lại nữa.

        Lưu ý:
            Không gọi hàm này cho toàn bộ constraints một cách bừa bãi.
            Chỉ gọi với selected_constraint hoặc active_constraint hiện tại.
        """

        if constraint is None:
            return None

        if not self.is_hard(
            constraint
        ):
            return constraint

        target_phrase = getattr(
            constraint,
            "target_phrase",
            "",
        )

        if require_output_endswith_target:
            found = self.target_ends_output(
                target_phrase,
                generated_text,
            )
        else:
            found = self.target_exists_in_text(
                target_phrase,
                generated_text,
            )

        if found:
            return self.mark_done(
                constraint
            )

        return constraint

    def reset(
        self,
        constraint,
    ):
        """
        Đưa constraint về NOT_STARTED.
        """

        if constraint is None:
            return None

        constraint.state = self.NOT_STARTED

        fsa = getattr(
            constraint,
            "fsa",
            None,
        )

        if fsa is not None and hasattr(
            fsa,
            "reset",
        ):
            try:
                fsa.reset()
            except Exception:
                pass

        return constraint

    def clear_active_if_done(
        self,
        constraints,
    ):
        """
        Nếu constraint ACTIVE có FSA đã DONE,
        chuyển sang DONE.
        """

        if not constraints:
            return constraints

        for constraint in constraints:
            if self.is_active(
                constraint
            ):
                self.mark_done_if_fsa_done(
                    constraint
                )

        return constraints

    # --------------------------------------------------
    # Satisfied by generated text
    # --------------------------------------------------

    def mark_satisfied_by_text(
        self,
        constraints,
        generated_text: str,
    ):
        """
        Cập nhật thông tin text_found cho debug/validator.

        SOFT:
            target_found=True/False

        HARD:
            chỉ ghi target_found, không tự mark DONE toàn cục.
        """

        if not constraints:
            return constraints

        for constraint in constraints:
            target_phrase = getattr(
                constraint,
                "target_phrase",
                "",
            )

            found = self.target_exists_in_text(
                target_phrase,
                generated_text,
            )

            setattr(
                constraint,
                "target_found",
                found,
            )

            if self.is_soft(
                constraint
            ):
                setattr(
                    constraint,
                    "soft_satisfied",
                    found,
                )

        return constraints

    def mark_satisfied_by_active_fsa(
        self,
        constraint,
    ):
        """
        Cách an toàn:
            ACTIVE constraint chỉ DONE khi FSA DONE.
        """

        return self.mark_done_if_fsa_done(
            constraint
        )

    # --------------------------------------------------
    # Rewrite support
    # --------------------------------------------------

    def mark_rewritten_done(
        self,
        constraint,
    ):
        """
        Chỉ dùng nếu có cơ chế rollback/rewrite.
        """

        return self.mark_done(
            constraint
        )

    # --------------------------------------------------
    # Debug / progress
    # --------------------------------------------------

    def progress(
        self,
        constraints,
    ):
        if not constraints:
            return {
                "total": 0,
                "hard_total": 0,
                "soft_total": 0,
                "pending": 0,
                "active": 0,
                "done": 0,
                "hard_done": 0,
                "soft": 0,
                "soft_found": 0,
                "soft_missing": 0,
            }

        total = len(
            constraints
        )

        hard_constraints = [
            constraint
            for constraint in constraints
            if self.is_hard(
                constraint
            )
        ]

        soft_constraints = [
            constraint
            for constraint in constraints
            if self.is_soft(
                constraint
            )
        ]

        active = len(
            [
                constraint
                for constraint in constraints
                if self.is_active(
                    constraint
                )
            ]
        )

        done = len(
            [
                constraint
                for constraint in constraints
                if self.is_done(
                    constraint
                )
            ]
        )

        pending = len(
            [
                constraint
                for constraint in hard_constraints
                if self.is_not_started(
                    constraint
                )
            ]
        )

        hard_done = len(
            [
                constraint
                for constraint in hard_constraints
                if self.is_done(
                    constraint
                )
            ]
        )

        soft_found = len(
            [
                constraint
                for constraint in soft_constraints
                if bool(
                    getattr(
                        constraint,
                        "soft_satisfied",
                        False,
                    )
                    or getattr(
                        constraint,
                        "target_found",
                        False,
                    )
                )
            ]
        )

        soft_missing = max(
            0,
            len(
                soft_constraints
            )
            - soft_found,
        )

        return {
            "total": total,
            "hard_total": len(
                hard_constraints
            ),
            "soft_total": len(
                soft_constraints
            ),
            "pending": pending,
            "active": active,
            "done": done,
            "hard_done": hard_done,
            "soft": len(
                soft_constraints
            ),
            "soft_found": soft_found,
            "soft_missing": soft_missing,
        }

    def constraint_to_text(
        self,
        constraint,
        generated_text: str = "",
    ) -> str:
        if constraint is None:
            return "None"

        status = self.display_status(
            constraint,
            generated_text,
        )

        fsa = getattr(
            constraint,
            "fsa",
            None,
        )

        if fsa is None:
            fsa_text = "fsa=None"
        else:
            fsa_text = (
                f"fsa_mode={getattr(fsa, 'mode', None)}, "
                f"phase={getattr(fsa, 'phase', None)}, "
                f"pos={getattr(fsa, 'position', None)}, "
                f"done={getattr(fsa, 'is_done', None)}"
            )

        return (
            f"{getattr(constraint, 'source_phrase', '')} -> "
            f"{getattr(constraint, 'target_phrase', '')} "
            f"[{status}] "
            f"state={getattr(constraint, 'state', None)} "
            f"force={getattr(constraint, 'force', None)} "
            f"type={getattr(constraint, 'constraint_type', None)} "
            f"protect={getattr(constraint, 'protect', None)} "
            f"{fsa_text}"
        )

    def constraint_to_dict(
        self,
        constraint,
        generated_text: str = "",
    ):
        if constraint is None:
            return None

        target_phrase = getattr(
            constraint,
            "target_phrase",
            "",
        )

        target_found = self.target_exists_in_text(
            target_phrase,
            generated_text,
        )

        status = self.display_status(
            constraint,
            generated_text,
        )

        return {
            "source_phrase": getattr(
                constraint,
                "source_phrase",
                None,
            ),
            "target_phrase": target_phrase,
            "status": status,
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
            "protect": getattr(
                constraint,
                "protect",
                None,
            ),
            "is_soft": self.is_soft(
                constraint
            ),
            "is_hard": self.is_hard(
                constraint
            ),
            "is_relaxed_hard": self.is_relaxed_hard(
                constraint
            ),
            "is_strict_hard": self.is_strict_hard(
                constraint
            ),
            "target_found": target_found,
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
            "fsa": self.fsa_debug(
                constraint
            ),
        }

    def constraints_to_dict(
        self,
        constraints,
        generated_text: str = "",
    ):
        return [
            self.constraint_to_dict(
                constraint,
                generated_text,
            )
            for constraint in constraints or []
        ]

    def print_status(
        self,
        constraints,
        generated_text: str = "",
    ):
        print()
        print("Constraint Status")
        print("---------------------")

        if not constraints:
            print("No constraints")
            print()
            return

        for constraint in constraints:
            print(
                self.constraint_to_text(
                    constraint,
                    generated_text,
                )
            )

        print()