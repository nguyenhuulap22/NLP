from collections import defaultdict
from typing import Dict, Tuple, List, Optional

from decoding.dba_beam import Beam
from decoding.hypothesis import Hypothesis


class MultiStackBeam:
    """
    Multi-stack Beam Search.

    Mỗi stack tương ứng với một trạng thái constraint khác nhau.

    Stack key dựa trên FSA position của từng constraint.

    Ví dụ với 1 constraint có target 2 token:
        stack key = (0,)  -> chưa sinh token nào của constraint
        stack key = (1,)  -> đã sinh token thứ nhất
        stack key = (2,)  -> constraint đã DONE

    Bản sửa:
        1. best() không chọn theo score thuần nữa.
        2. all_ended() không xem hypothesis là kết thúc hợp lệ
           nếu constraint chưa DONE.
    """

    def __init__(
        self,
        beam_size: int = 5,
        length_penalty: float = 1.0,
    ):
        self.beam_size = beam_size
        self.length_penalty = length_penalty

        self.stacks: Dict[Tuple[int, ...], Beam] = defaultdict(
            lambda: Beam(
                beam_size=self.beam_size,
                length_penalty=self.length_penalty,
            )
        )

    # --------------------------------------------------
    # Constraint helpers
    # --------------------------------------------------

    def _constraint_state(
        self,
        constraint,
    ):
        return getattr(
            constraint,
            "state",
            None,
        )

    def _constraint_done_count(
        self,
        hyp: Hypothesis,
    ) -> int:
        if not hyp.constraints:
            return 0

        return sum(
            1
            for constraint in hyp.constraints
            if self._constraint_state(
                constraint
            )
            == "DONE"
        )

    def _constraint_active_count(
        self,
        hyp: Hypothesis,
    ) -> int:
        if not hyp.constraints:
            return 0

        return sum(
            1
            for constraint in hyp.constraints
            if self._constraint_state(
                constraint
            )
            == "ACTIVE"
        )

    def _constraint_total(
        self,
        hyp: Hypothesis,
    ) -> int:
        if not hyp.constraints:
            return 0

        return len(
            hyp.constraints
        )

    def _all_constraints_done(
        self,
        hyp: Hypothesis,
    ) -> bool:
        if not hyp.constraints:
            return True

        return self._constraint_done_count(
            hyp
        ) == self._constraint_total(
            hyp
        )

    def _fsa_progress_sum(
        self,
        hyp: Hypothesis,
    ) -> int:
        if not hyp.constraints:
            return 0

        total = 0

        for constraint in hyp.constraints:
            fsa = getattr(
                constraint,
                "fsa",
                None,
            )

            if fsa is None:
                continue

            total += int(
                getattr(
                    fsa,
                    "position",
                    0,
                )
            )

        return total

    # --------------------------------------------------
    # Stack key
    # --------------------------------------------------

    def progress_key(
        self,
        hyp: Hypothesis,
    ) -> Tuple[int, ...]:
        """
        Tạo stack key dựa trên vị trí FSA của từng constraint.

        Nếu constraint chưa có FSA:
            dùng 0

        Nếu constraint có FSA:
            dùng constraint.fsa.position
        """

        key = []

        for constraint in hyp.constraints:
            fsa = getattr(
                constraint,
                "fsa",
                None,
            )

            if fsa is None:
                key.append(
                    0
                )
            else:
                key.append(
                    int(
                        getattr(
                            fsa,
                            "position",
                            0,
                        )
                    )
                )

        return tuple(
            key
        )

    # --------------------------------------------------
    # Beam operations
    # --------------------------------------------------

    def add(
        self,
        hyp: Hypothesis,
    ) -> None:
        """
        Thêm hypothesis vào stack tương ứng.
        """

        key = self.progress_key(
            hyp
        )

        self.stacks[key].add(
            hyp
        )

    def extend(
        self,
        hyps: List[Hypothesis],
    ) -> None:
        """
        Thêm nhiều hypothesis vào multi-stack.
        """

        for hyp in hyps:
            self.add(
                hyp
            )

    def prune(
        self,
    ) -> None:
        """
        Prune từng stack riêng biệt.

        Đây là điểm khác beam search thường:
            mỗi trạng thái constraint được giữ beam riêng.
        """

        for beam in self.stacks.values():
            beam.prune()

    def all_hypotheses(
        self,
    ) -> List[Hypothesis]:
        """
        Lấy tất cả hypothesis từ mọi stack.
        """

        result = []

        for beam in self.stacks.values():
            result.extend(
                beam.hypotheses
            )

        return result

    # --------------------------------------------------
    # Best hypothesis
    # --------------------------------------------------

    def best(
        self,
    ) -> Optional[Hypothesis]:
        """
        Lấy hypothesis tốt nhất trên toàn bộ stack.

        Không chọn theo score thuần.

        Ưu tiên:
            1. Constraint DONE nhiều hơn.
            2. FSA progress cao hơn.
            3. ACTIVE ít hơn.
            4. Normalized score cao hơn.
        """

        hyps = self.all_hypotheses()

        if not hyps:
            return None

        return max(
            hyps,
            key=lambda hyp: (
                self._constraint_done_count(
                    hyp
                ),
                self._fsa_progress_sum(
                    hyp
                ),
                -self._constraint_active_count(
                    hyp
                ),
                hyp.normalized_score(
                    self.length_penalty
                ),
            ),
        )

    # --------------------------------------------------
    # Stop condition
    # --------------------------------------------------

    def all_ended(
        self,
        require_constraints_done: bool = True,
    ) -> bool:
        """
        Kiểm tra tất cả hypothesis đã kết thúc hợp lệ chưa.

        Nếu require_constraints_done=True:
            một hypothesis chỉ xem là kết thúc hợp lệ khi:
                - hyp.ended == True
                - tất cả constraints DONE

        Điều này tránh việc decoder sinh EOS sớm
        trong khi constraint vẫn NOT_STARTED hoặc ACTIVE.
        """

        hyps = self.all_hypotheses()

        if not hyps:
            return False

        if not require_constraints_done:
            return all(
                hyp.ended
                for hyp in hyps
            )

        return all(
            hyp.ended and self._all_constraints_done(
                hyp
            )
            for hyp in hyps
        )

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------

    def clear(
        self,
    ) -> None:
        """
        Xóa toàn bộ stack.
        """

        self.stacks.clear()

    def stack_summary(
        self,
    ):
        """
        Tóm tắt số hypothesis trong từng stack.
        Dùng để debug UI.
        """

        return {
            str(key): len(
                beam.hypotheses
            )
            for key, beam in self.stacks.items()
        }