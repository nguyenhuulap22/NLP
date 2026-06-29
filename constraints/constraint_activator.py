from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from decoding.attention_monitor import AttentionInfo
from terminology.constraint import (
    Constraint,
    ConstraintType,
    priority_score,
)


@dataclass
class ActivationDecision:
    """
    Kết quả kích hoạt constraint tại một decoding step.
    """

    activated: bool = False
    continued: bool = False
    finished_by_sync: bool = False

    constraint: Optional[Constraint] = None
    constraint_id: Optional[str] = None

    reason: str = "no_activation"

    focus_pos: Optional[int] = None
    focus_score: float = 0.0
    span_score: float = 0.0

    trigger_type: str = "none"

    candidates: List[Dict[str, Any]] = field(
        default_factory=list
    )

    def has_constraint(self) -> bool:
        return self.constraint is not None

    def to_dict(
        self,
        compact: bool = True,
    ) -> Dict[str, Any]:
        data = {
            "activated": self.activated,
            "continued": self.continued,
            "finished_by_sync": self.finished_by_sync,
            "constraint_id": self.constraint_id,
            "reason": self.reason,
            "trigger_type": self.trigger_type,
            "focus_pos": self.focus_pos,
            "focus_score": self.focus_score,
            "span_score": self.span_score,
        }

        if self.constraint is not None:
            constraint_type = getattr(
                self.constraint,
                "constraint_type",
                None,
            )

            state = getattr(
                self.constraint,
                "state",
                None,
            )

            data.update(
                {
                    "source_phrase": getattr(
                        self.constraint,
                        "source_phrase",
                        "",
                    ),
                    "target_phrase": getattr(
                        self.constraint,
                        "target_phrase",
                        "",
                    ),
                    "constraint_type": constraint_type.value
                    if hasattr(constraint_type, "value")
                    else str(constraint_type),
                    "state": state.value
                    if hasattr(state, "value")
                    else str(state),
                    "covered": bool(
                        getattr(
                            self.constraint,
                            "covered",
                            False,
                        )
                    ),
                }
            )

        if not compact:
            data["candidates"] = list(
                self.candidates
            )

        return data


class ConstraintActivator:
    """
    Kích hoạt constraint bằng cross-attention.

    Vai trò:
        1. Nếu đang có ACTIVE constraint:
            tiếp tục constraint đó.

        2. Nếu chưa có ACTIVE constraint:
            dùng attention để tìm source span phù hợp.

        3. Nếu attention chạm đúng span:
            activate constraint.

        4. Nếu output đã sinh một phần target phrase:
            sync FSA để không ép lại từ đầu.

        5. Nếu FSA DONE:
            mark_done + covered=True.

    Không làm:
        - build FSA
        - mask logits
        - chọn token
        - replace output sau dịch
    """

    def __init__(
        self,
        min_focus_score: float = 0.10,
        min_span_score: float = 0.25,
        use_topk_intersection: bool = False,
        allow_protected: bool = True,

        # Secondary attention:
        # dùng khi focus_pos không nằm trong span,
        # nhưng tổng attention trên span vẫn đủ mạnh.
        secondary_span_score: float = 0.18,
        secondary_topk_score: float = 0.18,
    ):
        self.min_focus_score = float(
            min_focus_score
        )

        self.min_span_score = float(
            min_span_score
        )

        self.use_topk_intersection = bool(
            use_topk_intersection
        )

        self.allow_protected = bool(
            allow_protected
        )

        self.secondary_span_score = float(
            secondary_span_score
        )

        self.secondary_topk_score = float(
            secondary_topk_score
        )

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def activate(
        self,
        constraints: List[Constraint],
        attention_info: Optional[AttentionInfo],
        generated_token_ids: Optional[Sequence[int]] = None,
        step: Optional[int] = None,
    ) -> ActivationDecision:
        generated_token_ids = list(
            generated_token_ids or []
        )

        active = self._active_constraint(
            constraints
        )

        if active is not None:
            finished = self._sync_or_finish_active(
                active,
                generated_token_ids,
                step=step,
            )

            return ActivationDecision(
                activated=False,
                continued=not finished,
                finished_by_sync=finished,
                constraint=active,
                constraint_id=getattr(
                    active,
                    "id",
                    None,
                ),
                reason="active_constraint_finished_by_fsa"
                if finished
                else "continue_active_constraint",
                trigger_type="active",
                focus_pos=attention_info.focus_pos
                if attention_info is not None
                else None,
                focus_score=attention_info.focus_score
                if attention_info is not None
                else 0.0,
                span_score=attention_info.span_score(
                    active.token_span
                )
                if attention_info is not None
                and getattr(active, "token_span", None) is not None
                else 0.0,
            )

        if attention_info is None or not getattr(
            attention_info,
            "valid",
            False,
        ):
            return ActivationDecision(
                activated=False,
                continued=False,
                reason="invalid_attention",
            )

        candidates = self._matching_candidates(
            constraints=constraints,
            attention_info=attention_info,
        )

        if not candidates:
            return ActivationDecision(
                activated=False,
                continued=False,
                reason="no_constraint_matches_attention",
                focus_pos=attention_info.focus_pos,
                focus_score=attention_info.focus_score,
            )

        selected, selected_info = self._select_candidate(
            candidates
        )

        if selected is None:
            return ActivationDecision(
                activated=False,
                continued=False,
                reason="no_selected_candidate",
                focus_pos=attention_info.focus_pos,
                focus_score=attention_info.focus_score,
                candidates=[
                    info
                    for _, info in candidates
                ],
            )

        finished_by_sync = self._activate_constraint(
            constraint=selected,
            generated_token_ids=generated_token_ids,
            step=step,
            activation_info=selected_info,
        )

        return ActivationDecision(
            activated=not finished_by_sync,
            continued=False,
            finished_by_sync=finished_by_sync,
            constraint=selected,
            constraint_id=getattr(
                selected,
                "id",
                None,
            ),
            reason="constraint_already_completed_by_generated_tail"
            if finished_by_sync
            else "attention_triggered_activation",
            trigger_type=str(
                selected_info.get(
                    "trigger_type",
                    "attention",
                )
            ),
            focus_pos=attention_info.focus_pos,
            focus_score=attention_info.focus_score,
            span_score=float(
                selected_info.get(
                    "span_score",
                    0.0,
                )
            ),
            candidates=[
                info
                for _, info in candidates
            ],
        )

    # --------------------------------------------------
    # Active handling
    # --------------------------------------------------

    def _active_constraint(
        self,
        constraints: List[Constraint],
    ) -> Optional[Constraint]:
        for constraint in constraints or []:
            if not self._is_forceable(
                constraint
            ):
                continue

            if constraint.is_active():
                return constraint

        return None

    def _fsa_is_done(
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

        if callable(value):
            try:
                return bool(
                    value()
                )
            except Exception:
                return False

        return bool(value)

    def _sync_or_finish_active(
        self,
        constraint: Constraint,
        generated_token_ids: Sequence[int],
        step: Optional[int] = None,
    ) -> bool:
        self._sync_fsa_partial_prefix_with_generated_tail(
            constraint=constraint,
            generated_token_ids=generated_token_ids,
        )

        fsa = getattr(
            constraint,
            "fsa",
            None,
        )

        if fsa is not None and hasattr(
            fsa,
            "sync_with_generated_tail",
        ):
            try:
                fsa.sync_with_generated_tail(
                    generated_token_ids
                )
            except Exception:
                pass

        if self._fsa_is_done(
            constraint
        ):
            self._mark_done(
                constraint=constraint,
                step=step,
                reason="active_fsa_done",
            )
            return True

        return False

    # --------------------------------------------------
    # Forceable / matching
    # --------------------------------------------------

    def _is_forceable(
        self,
        constraint: Optional[Constraint],
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
            return False

        if constraint.is_done() or constraint.is_blocked():
            return False

        constraint_type = getattr(
            constraint,
            "constraint_type",
            ConstraintType.SOFT,
        )

        if constraint_type == ConstraintType.SOFT:
            return False

        if (
            constraint_type == ConstraintType.PROTECTED
            and not self.allow_protected
        ):
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

        if not constraint.has_token_span():
            return False

        return True

    def _matching_candidates(
        self,
        constraints: List[Constraint],
        attention_info: AttentionInfo,
    ) -> List[Tuple[Constraint, Dict[str, Any]]]:
        result: List[Tuple[Constraint, Dict[str, Any]]] = []

        for constraint in constraints or []:
            if not self._is_forceable(
                constraint
            ):
                continue

            if not constraint.is_pending():
                continue

            matched, info = self._matches_attention(
                constraint=constraint,
                attention_info=attention_info,
            )

            if matched:
                result.append(
                    (
                        constraint,
                        info,
                    )
                )

        return result

    def _matches_attention(
        self,
        constraint: Constraint,
        attention_info: AttentionInfo,
    ) -> Tuple[bool, Dict[str, Any]]:
        span = getattr(
            constraint,
            "token_span",
            None,
        )

        focus_in_span = attention_info.focus_in_span(
            span
        )

        span_score = attention_info.span_score(
            span
        )

        topk_intersects = attention_info.topk_intersects_span(
            span
        )

        focus_ok = (
            focus_in_span
            and attention_info.focus_score >= self.min_focus_score
        )

        span_ok = span_score >= self.min_span_score

        secondary_ok = (
            not focus_in_span
            and span_score >= self.secondary_span_score
        )

        topk_ok = (
            self.use_topk_intersection
            and topk_intersects
            and span_score >= self.secondary_topk_score
        )

        matched = bool(
            focus_ok
            or span_ok
            or secondary_ok
            or topk_ok
        )

        if focus_ok:
            trigger_type = "focus_in_span"
        elif span_ok:
            trigger_type = "span_score"
        elif secondary_ok:
            trigger_type = "secondary_attention"
        elif topk_ok:
            trigger_type = "topk_intersection"
        else:
            trigger_type = "none"

        info = {
            "constraint_id": constraint.id,
            "source_phrase": constraint.source_phrase,
            "target_phrase": constraint.target_phrase,
            "token_span": constraint.token_span,
            "source_order": constraint.source_order,

            "focus_pos": attention_info.focus_pos,
            "focus_score": attention_info.focus_score,
            "focus_in_span": focus_in_span,

            "span_score": span_score,
            "topk_intersects": topk_intersects,

            "focus_ok": focus_ok,
            "span_ok": span_ok,
            "secondary_ok": secondary_ok,
            "topk_ok": topk_ok,
            "matched": matched,
            "trigger_type": trigger_type,
        }

        return matched, info

    # --------------------------------------------------
    # Candidate selection
    # --------------------------------------------------

    def _select_candidate(
        self,
        candidates: List[Tuple[Constraint, Dict[str, Any]]],
    ) -> Tuple[Optional[Constraint], Dict[str, Any]]:
        if not candidates:
            return None, {}

        sorted_candidates = sorted(
            candidates,
            key=lambda item: self._candidate_score(
                item[0],
                item[1],
            ),
            reverse=True,
        )

        return sorted_candidates[0]

    def _candidate_score(
        self,
        constraint: Constraint,
        info: Dict[str, Any],
    ):
        constraint_type_score = {
            ConstraintType.SOFT: 1,
            ConstraintType.HARD: 2,
            ConstraintType.PROTECTED: 3,
        }.get(
            constraint.constraint_type,
            1,
        )

        phrase_len = len(
            str(
                constraint.source_phrase
            ).split()
        )

        trigger_score = {
            "focus_in_span": 4,
            "span_score": 3,
            "secondary_attention": 2,
            "topk_intersection": 1,
            "none": 0,
        }.get(
            str(
                info.get(
                    "trigger_type",
                    "none",
                )
            ),
            0,
        )

        return (
            trigger_score,
            float(
                info.get(
                    "span_score",
                    0.0,
                )
            ),
            1
            if info.get(
                "focus_in_span",
                False,
            )
            else 0,
            priority_score(
                constraint.priority
            ),
            constraint_type_score,
            phrase_len,
            -int(
                getattr(
                    constraint,
                    "source_order",
                    999999,
                )
            ),
        )

    # --------------------------------------------------
    # Activation
    # --------------------------------------------------

    def _activate_constraint(
        self,
        constraint: Constraint,
        generated_token_ids: Sequence[int],
        step: Optional[int] = None,
        activation_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Return:
            True:
                constraint đã DONE ngay sau sync.

            False:
                constraint thật sự ACTIVE.
        """

        self._sync_fsa_partial_prefix_with_generated_tail(
            constraint=constraint,
            generated_token_ids=generated_token_ids,
        )

        fsa = getattr(
            constraint,
            "fsa",
            None,
        )

        if fsa is not None and hasattr(
            fsa,
            "sync_with_generated_tail",
        ):
            try:
                fsa.sync_with_generated_tail(
                    generated_token_ids
                )
            except Exception:
                pass

        if self._fsa_is_done(
            constraint
        ):
            self._mark_done(
                constraint=constraint,
                step=step,
                reason="already_generated_tail",
                activation_info=activation_info,
            )
            return True

        constraint.activate()

        if self._fsa_is_done(
            constraint
        ):
            self._mark_done(
                constraint=constraint,
                step=step,
                reason="activated_then_done",
                activation_info=activation_info,
            )
            return True

        meta = self._meta(
            constraint
        )

        meta["activated_step"] = step
        meta["activation_reason"] = "attention_focus_or_span_score"

        if activation_info:
            meta["activation_info"] = dict(
                activation_info
            )

        return False

    def _mark_done(
        self,
        constraint: Constraint,
        step: Optional[int] = None,
        reason: str = "done",
        activation_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            constraint.mark_done()
        except Exception:
            try:
                constraint.state = "DONE"
            except Exception:
                pass

        try:
            constraint.covered = True
        except Exception:
            pass

        meta = self._meta(
            constraint
        )

        meta["done_step"] = step
        meta["activation_reason"] = reason
        meta["covered_by_activator"] = True

        if activation_info:
            meta["activation_info"] = dict(
                activation_info
            )

    # --------------------------------------------------
    # Partial prefix sync
    # --------------------------------------------------

    def _target_token_ids(
        self,
        constraint: Constraint,
    ) -> List[int]:
        target_ids = getattr(
            constraint,
            "target_token_ids",
            None,
        )

        if not target_ids:
            fsa = getattr(
                constraint,
                "fsa",
                None,
            )

            target_ids = getattr(
                fsa,
                "target_token_ids",
                None,
            )

        if not target_ids:
            return []

        try:
            return [
                int(
                    token_id
                )
                for token_id in target_ids
            ]
        except Exception:
            return []

    def _sync_fsa_partial_prefix_with_generated_tail(
        self,
        constraint: Optional[Constraint],
        generated_token_ids: Sequence[int],
    ) -> None:
        if constraint is None:
            return

        fsa = getattr(
            constraint,
            "fsa",
            None,
        )

        if fsa is None:
            return

        target_ids = self._target_token_ids(
            constraint
        )

        if not target_ids or not generated_token_ids:
            return

        try:
            generated_ids = [
                int(
                    token_id
                )
                for token_id in generated_token_ids
            ]
        except Exception:
            return

        current_pos = int(
            getattr(
                fsa,
                "position",
                0,
            )
            or 0
        )

        best_pos = current_pos

        max_len = min(
            len(
                target_ids
            ),
            len(
                generated_ids
            ),
        )

        for n in range(
            1,
            max_len + 1,
        ):
            if generated_ids[
                -n:
            ] == target_ids[
                :n
            ]:
                best_pos = max(
                    best_pos,
                    n,
                )

        if best_pos <= current_pos:
            return

        try:
            fsa.position = best_pos
        except Exception:
            return

        if best_pos >= len(
            target_ids
        ):
            self._mark_done(
                constraint=constraint,
                reason="partial_prefix_completed",
            )

    # --------------------------------------------------
    # Meta helper
    # --------------------------------------------------

    def _meta(
        self,
        constraint: Constraint,
    ) -> Dict[str, Any]:
        meta = getattr(
            constraint,
            "meta",
            None,
        )

        if isinstance(
            meta,
            dict,
        ):
            return meta

        try:
            constraint.meta = {}
            return constraint.meta
        except Exception:
            return {}

    # --------------------------------------------------
    # Helpers for decoding loop
    # --------------------------------------------------

    def active_allowed_token_ids(
        self,
        constraints: List[Constraint],
    ) -> List[int]:
        active = self._active_constraint(
            constraints
        )

        if active is None:
            return []

        return active.allowed_token_ids()

    def mark_done_if_fsa_done(
        self,
        constraints: List[Constraint],
    ) -> None:
        for constraint in constraints or []:
            fsa = getattr(
                constraint,
                "fsa",
                None,
            )

            if fsa is None:
                continue

            if self._fsa_is_done(
                constraint
            ):
                self._mark_done(
                    constraint=constraint,
                    reason="mark_done_if_fsa_done",
                )