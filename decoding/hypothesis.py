from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import copy

from terminology.constraint import (
    Constraint,
    ConstraintType,
    ConstraintState,
)


@dataclass
class Hypothesis:
    """
    Một hypothesis trong beam search.

    Mỗi hypothesis lưu:

        token_ids:
            token đã sinh, bao gồm decoder_start_token_id.

        score:
            tổng log probability.

        constraints:
            danh sách Constraint riêng của hypothesis.

        trace:
            debug trace ngắn.

        ended:
            True nếu đã sinh EOS.

    Nguyên tắc rất quan trọng:

        Mỗi hypothesis phải có bản clone riêng của Constraint/FSA.

    Nếu nhiều hypothesis dùng chung FSA object, khi một hypothesis
    sinh token constraint thì các hypothesis khác cũng bị đổi FSA state.
    Khi đó beam search sẽ sai.
    """

    token_ids: List[int]
    score: float = 0.0

    constraints: List[Constraint] = field(
        default_factory=list
    )

    trace: List[Any] = field(
        default_factory=list
    )

    ended: bool = False

    # --------------------------------------------------
    # Basic helpers
    # --------------------------------------------------

    def last_token_id(
        self,
    ) -> Optional[int]:
        if not self.token_ids:
            return None

        return int(
            self.token_ids[
                -1
            ]
        )

    def length(
        self,
    ) -> int:
        """
        Độ dài thật của output.

        Không tính decoder_start_token_id.
        """

        return max(
            0,
            len(
                self.token_ids
            )
            - 1,
        )

    def is_empty(
        self,
    ) -> bool:
        return len(
            self.token_ids
        ) == 0

    def decoder_input_ids(
        self,
    ) -> List[int]:
        return [
            int(
                token_id
            )
            for token_id in self.token_ids
        ]

    # --------------------------------------------------
    # Constraint helpers
    # --------------------------------------------------

    def forced_constraints(
        self,
    ) -> List[Constraint]:
        return [
            constraint
            for constraint in self.constraints
            if self._is_forceable_constraint(
                constraint
            )
        ]

    def active_constraint(
        self,
    ) -> Optional[Constraint]:
        for constraint in self.forced_constraints():
            if self._constraint_is_active(
                constraint
            ):
                return constraint

        return None

    def pending_forced_constraints(
        self,
    ) -> List[Constraint]:
        return [
            constraint
            for constraint in self.forced_constraints()
            if self._constraint_is_pending(
                constraint
            )
        ]

    def done_forced_constraints(
        self,
    ) -> List[Constraint]:
        return [
            constraint
            for constraint in self.forced_constraints()
            if self._constraint_is_done(
                constraint
            )
        ]

    def forced_done_count(
        self,
    ) -> int:
        return len(
            self.done_forced_constraints()
        )

    def forced_total_count(
        self,
    ) -> int:
        return len(
            self.forced_constraints()
        )

    def all_forced_done(
        self,
    ) -> bool:
        forced = self.forced_constraints()

        if not forced:
            return True

        return all(
            self._constraint_is_done(
                constraint
            )
            for constraint in forced
        )

    def has_active_constraint(
        self,
    ) -> bool:
        return self.active_constraint() is not None

    # --------------------------------------------------
    # Constraint state checks
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

    def _is_forceable_constraint(
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

    def _constraint_is_active(
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

        return self._state_value(
            constraint
        ).upper() == "ACTIVE"

    def _constraint_is_pending(
        self,
        constraint: Constraint,
    ) -> bool:
        if hasattr(
            constraint,
            "is_pending",
        ):
            try:
                return bool(
                    constraint.is_pending()
                )
            except Exception:
                pass

        return self._state_value(
            constraint
        ).upper() == "PENDING"

    def _constraint_is_done(
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

        if self._state_value(
            constraint
        ).upper() == "DONE":
            return True

        fsa = getattr(
            constraint,
            "fsa",
            None,
        )

        if fsa is not None and bool(
            getattr(
                fsa,
                "is_done",
                False,
            )
        ):
            return True

        return False

    # --------------------------------------------------
    # FSA stepping
    # --------------------------------------------------

    def step_active_constraint(
        self,
        token_id: int,
    ) -> bool:
        """
        Sau khi hypothesis sinh token_id, cập nhật FSA của constraint ACTIVE.

        Return:
            True:
                token hợp lệ hoặc không có active constraint.

            False:
                token không hợp lệ với active FSA.
                Candidate này nên bị loại khỏi beam.
        """

        active = self.active_constraint()

        if active is None:
            return True

        fsa = getattr(
            active,
            "fsa",
            None,
        )

        if fsa is None:
            return True

        if not hasattr(
            fsa,
            "step",
        ):
            return True

        ok = bool(
            fsa.step(
                int(
                    token_id
                )
            )
        )

        if not ok:
            if hasattr(
                fsa,
                "fail",
            ):
                try:
                    fsa.fail()
                except Exception:
                    pass

            return False

        if bool(
            getattr(
                fsa,
                "is_done",
                False,
            )
        ):
            if hasattr(
                active,
                "mark_done",
            ):
                active.mark_done()
            else:
                active.state = ConstraintState.DONE

        return True

    def sync_done_constraints(
        self,
    ) -> None:
        """
        Đồng bộ Constraint.state với FSA state.

        Nếu fsa.is_done=True thì constraint.mark_done().
        """

        for constraint in self.forced_constraints():
            fsa = getattr(
                constraint,
                "fsa",
                None,
            )

            if fsa is None:
                continue

            if bool(
                getattr(
                    fsa,
                    "is_done",
                    False,
                )
            ):
                if hasattr(
                    constraint,
                    "mark_done",
                ):
                    constraint.mark_done()
                else:
                    constraint.state = ConstraintState.DONE

    # --------------------------------------------------
    # Clone
    # --------------------------------------------------

    def _clone_one_constraint(
        self,
        constraint: Constraint,
    ) -> Constraint:
        if constraint is None:
            return None

        if hasattr(
            constraint,
            "clone",
        ):
            try:
                return constraint.clone()
            except Exception:
                pass

        try:
            return copy.deepcopy(
                constraint
            )
        except Exception:
            pass

        try:
            return copy.copy(
                constraint
            )
        except Exception:
            return constraint

    def _clone_constraints(
        self,
    ) -> List[Constraint]:
        if not self.constraints:
            return []

        return [
            self._clone_one_constraint(
                constraint
            )
            for constraint in self.constraints
        ]

    def _clone_trace(
        self,
    ) -> List[Any]:
        if not self.trace:
            return []

        try:
            return copy.deepcopy(
                self.trace
            )
        except Exception:
            return list(
                self.trace
            )

    def clone(
        self,
    ) -> "Hypothesis":
        return Hypothesis(
            token_ids=[
                int(
                    token_id
                )
                for token_id in self.token_ids
            ],
            score=float(
                self.score
            ),
            constraints=self._clone_constraints(),
            trace=self._clone_trace(),
            ended=bool(
                self.ended
            ),
        )

    # --------------------------------------------------
    # Extend
    # --------------------------------------------------

    def extend(
        self,
        token_id: int,
        log_prob: float,
        eos_token_id: Optional[int] = None,
        trace_item: Any = None,
        step_constraints: bool = True,
    ) -> Optional["Hypothesis"]:
        """
        Tạo hypothesis mới bằng cách thêm token.

        Nếu step_constraints=True:
            sau khi thêm token, hypothesis sẽ cập nhật FSA ACTIVE.

        Nếu token không hợp lệ với FSA:
            return None
        """

        new_hyp = self.clone()

        token_id = int(
            token_id
        )

        new_hyp.token_ids.append(
            token_id
        )

        new_hyp.score += float(
            log_prob
        )

        if step_constraints:
            ok = new_hyp.step_active_constraint(
                token_id
            )

            if not ok:
                return None

        if eos_token_id is not None and token_id == int(
            eos_token_id
        ):
            new_hyp.ended = True

        if trace_item is not None:
            try:
                new_hyp.trace.append(
                    copy.deepcopy(
                        trace_item
                    )
                )
            except Exception:
                new_hyp.trace.append(
                    trace_item
                )

        return new_hyp

    def append_trace(
        self,
        trace_item: Any,
    ) -> None:
        if trace_item is None:
            return

        try:
            self.trace.append(
                copy.deepcopy(
                    trace_item
                )
            )
        except Exception:
            self.trace.append(
                trace_item
            )

    # --------------------------------------------------
    # Score
    # --------------------------------------------------

    def normalized_score(
        self,
        length_penalty: float = 1.0,
    ) -> float:
        """
        Score chuẩn hóa theo độ dài.

        Không tính decoder_start_token_id.
        """

        length = max(
            1,
            self.length(),
        )

        length_penalty = float(
            length_penalty
        )

        if length_penalty == 0:
            return float(
                self.score
            )

        if length_penalty == 1.0:
            return float(
                self.score
            ) / length

        return float(
            self.score
        ) / (
            length ** length_penalty
        )

    def final_score(
        self,
        length_penalty: float = 1.0,
        constraint_bonus: float = 0.0,
    ) -> float:
        """
        Score dùng khi chọn hypothesis cuối.

        Có thể cộng bonus nhỏ cho hypothesis hoàn thành nhiều constraints.
        """

        score = self.normalized_score(
            length_penalty=length_penalty
        )

        if constraint_bonus:
            score += float(
                constraint_bonus
            ) * self.forced_done_count()

        return score

    # --------------------------------------------------
    # DBA / bank key
    # --------------------------------------------------

    def constraint_progress_key(
        self,
    ) -> Tuple[int, int, Tuple[int, ...]]:
        """
        Key dùng cho DBA/multi-stack.

        Return:
            (
                số forced constraint DONE,
                có ACTIVE constraint hay không,
                tuple FSA positions
            )
        """

        forced = self.forced_constraints()

        done_count = sum(
            1
            for constraint in forced
            if self._constraint_is_done(
                constraint
            )
        )

        has_active = 1 if any(
            self._constraint_is_active(
                constraint
            )
            for constraint in forced
        ) else 0

        positions = []

        for constraint in forced:
            fsa = getattr(
                constraint,
                "fsa",
                None,
            )

            positions.append(
                int(
                    getattr(
                        fsa,
                        "position",
                        0,
                    )
                )
                if fsa is not None
                else 0
            )

        return (
            done_count,
            has_active,
            tuple(
                positions
            ),
        )

    def bank_id(
        self,
    ) -> int:
        """
        Bank đơn giản:
            số forced constraints đã DONE.

        DBA có thể dùng bank_id để phân bổ beam.
        """

        return self.forced_done_count()

    # --------------------------------------------------
    # Debug
    # --------------------------------------------------

    def constraints_summary(
        self,
    ) -> List[Dict[str, Any]]:
        result = []

        for constraint in self.constraints:
            fsa = getattr(
                constraint,
                "fsa",
                None,
            )

            result.append(
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
                    "constraint_type": self._constraint_type_value(
                        constraint
                    ),
                    "state": self._state_value(
                        constraint
                    ),
                    "force": getattr(
                        constraint,
                        "force",
                        False,
                    ),
                    "token_span": getattr(
                        constraint,
                        "token_span",
                        None,
                    ),
                    "has_fsa": fsa is not None,
                    "fsa_position": int(
                        getattr(
                            fsa,
                            "position",
                            0,
                        )
                    )
                    if fsa is not None
                    else None,
                    "fsa_done": bool(
                        getattr(
                            fsa,
                            "is_done",
                            False,
                        )
                    )
                    if fsa is not None
                    else False,
                }
            )

        return result

    def to_dict(
        self,
        include_constraints: bool = False,
        include_trace: bool = False,
    ) -> Dict[str, Any]:
        data = {
            "token_ids": list(
                self.token_ids
            ),
            "score": float(
                self.score
            ),
            "length": self.length(),
            "ended": bool(
                self.ended
            ),
            "forced_done": self.forced_done_count(),
            "forced_total": self.forced_total_count(),
            "has_active_constraint": self.has_active_constraint(),
            "constraint_progress_key": self.constraint_progress_key(),
            "trace_len": len(
                self.trace
            )
            if self.trace
            else 0,
        }

        if include_constraints:
            data[
                "constraints"
            ] = self.constraints_summary()

        if include_trace:
            data[
                "trace"
            ] = list(
                self.trace
            )

        return data