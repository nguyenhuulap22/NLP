from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, Iterator, List, Optional, Tuple

from decoding.hypothesis import Hypothesis


class Beam:
    """
    Beam thường: giữ top-k hypothesis tốt nhất.

    Class này giữ lại để tương thích với code cũ.
    Nếu cần constrained decoding, nên dùng DBABeam.
    """

    def __init__(
        self,
        beam_size: int = 5,
        length_penalty: float = 1.0,
        constraint_bonus: float = 0.0,
    ):
        self.beam_size = max(
            1,
            int(
                beam_size
            ),
        )

        self.length_penalty = float(
            length_penalty
        )

        self.constraint_bonus = float(
            constraint_bonus
        )

        self.hypotheses: List[Hypothesis] = []

    def _score(
        self,
        hyp: Hypothesis,
    ) -> float:
        if hasattr(
            hyp,
            "final_score",
        ):
            return float(
                hyp.final_score(
                    length_penalty=self.length_penalty,
                    constraint_bonus=self.constraint_bonus,
                )
            )

        return float(
            hyp.normalized_score(
                self.length_penalty
            )
        )

    def add(
        self,
        hyp: Optional[Hypothesis],
    ) -> None:
        if hyp is None:
            return

        self.hypotheses.append(
            hyp
        )

    def extend(
        self,
        hyps: Iterable[Hypothesis],
    ) -> None:
        if not hyps:
            return

        for hyp in hyps:
            self.add(
                hyp
            )

    def prune(
        self,
    ) -> None:
        self.hypotheses = sorted(
            self.hypotheses,
            key=self._score,
            reverse=True,
        )[
            : self.beam_size
        ]

    def topk(
        self,
        k: Optional[int] = None,
    ) -> List[Hypothesis]:
        k = self.beam_size if k is None else int(
            k
        )

        return sorted(
            self.hypotheses,
            key=self._score,
            reverse=True,
        )[
            :k
        ]

    def is_empty(
        self,
    ) -> bool:
        return len(
            self.hypotheses
        ) == 0

    def all_ended(
        self,
    ) -> bool:
        if not self.hypotheses:
            return False

        return all(
            hyp.ended
            for hyp in self.hypotheses
        )

    def best(
        self,
    ) -> Optional[Hypothesis]:
        if not self.hypotheses:
            return None

        return max(
            self.hypotheses,
            key=self._score,
        )

    def clear(
        self,
    ) -> None:
        self.hypotheses = []

    def __len__(
        self,
    ) -> int:
        return len(
            self.hypotheses
        )

    def __iter__(
        self,
    ) -> Iterator[Hypothesis]:
        return iter(
            self.hypotheses
        )


class DBABeam:
    """
    Dynamic Beam Allocation cho constrained decoding.

    Ý tưởng:

        Thay vì giữ top-k toàn cục,
        ta chia hypothesis thành các bank.

    Bank đơn giản:

        bank_id = số forced constraints đã DONE

    Ví dụ có 3 constraint:

        bank 0:
            chưa hoàn thành constraint nào

        bank 1:
            đã hoàn thành 1 constraint

        bank 2:
            đã hoàn thành 2 constraint

        bank 3:
            đã hoàn thành đủ 3 constraint

    Vì sao cần bank?

        Nếu chỉ sort theo score, hypothesis đang cố sinh constraint
        thường có score thấp hơn hypothesis dịch tự do.
        Nó dễ bị prune mất.
        DBA giữ một phần slot cho mỗi bank để các hypothesis
        có tiến độ constraint vẫn sống.

    File này KHÔNG:
        - gọi model
        - đọc attention
        - mask logits
        - step FSA

    File này CHỈ:
        - nhận danh sách Hypothesis
        - chia bank
        - prune theo beam allocation
    """

    def __init__(
        self,
        beam_size: int = 5,
        length_penalty: float = 1.0,
        constraint_bonus: float = 0.0,
        prefer_higher_banks: bool = True,
        deduplicate: bool = True,
    ):
        self.beam_size = max(
            1,
            int(
                beam_size
            ),
        )

        self.length_penalty = float(
            length_penalty
        )

        self.constraint_bonus = float(
            constraint_bonus
        )

        self.prefer_higher_banks = bool(
            prefer_higher_banks
        )

        self.deduplicate = bool(
            deduplicate
        )

        self.hypotheses: List[Hypothesis] = []

        self.last_debug: Dict[str, Any] = {}

    # --------------------------------------------------
    # Score
    # --------------------------------------------------

    def _score(
        self,
        hyp: Hypothesis,
    ) -> float:
        if hasattr(
            hyp,
            "final_score",
        ):
            return float(
                hyp.final_score(
                    length_penalty=self.length_penalty,
                    constraint_bonus=self.constraint_bonus,
                )
            )

        return float(
            hyp.normalized_score(
                self.length_penalty
            )
        )

    def _sort_hyps(
        self,
        hyps: Iterable[Hypothesis],
    ) -> List[Hypothesis]:
        return sorted(
            list(
                hyps
            ),
            key=self._score,
            reverse=True,
        )

    # --------------------------------------------------
    # Bank helpers
    # --------------------------------------------------

    def _bank_id(
        self,
        hyp: Hypothesis,
    ) -> int:
        if hasattr(
            hyp,
            "bank_id",
        ):
            try:
                return int(
                    hyp.bank_id()
                )
            except Exception:
                pass

        if hasattr(
            hyp,
            "forced_done_count",
        ):
            try:
                return int(
                    hyp.forced_done_count()
                )
            except Exception:
                pass

        return 0

    def _forced_total(
        self,
        hyp: Hypothesis,
    ) -> int:
        if hasattr(
            hyp,
            "forced_total_count",
        ):
            try:
                return int(
                    hyp.forced_total_count()
                )
            except Exception:
                pass

        return 0

    def _progress_key(
        self,
        hyp: Hypothesis,
    ) -> Tuple[Any, ...]:
        if hasattr(
            hyp,
            "constraint_progress_key",
        ):
            try:
                key = hyp.constraint_progress_key()

                if isinstance(
                    key,
                    tuple,
                ):
                    return key

                return (
                    key,
                )
            except Exception:
                pass

        return (
            self._bank_id(
                hyp
            ),
        )

    def _group_by_bank(
        self,
        hyps: Iterable[Hypothesis],
    ) -> Dict[int, List[Hypothesis]]:
        grouped: DefaultDict[int, List[Hypothesis]] = defaultdict(
            list
        )

        for hyp in hyps:
            grouped[
                self._bank_id(
                    hyp
                )
            ].append(
                hyp
            )

        return dict(
            grouped
        )

    # --------------------------------------------------
    # Deduplicate
    # --------------------------------------------------

    def _dedupe_key(
        self,
        hyp: Hypothesis,
    ) -> Tuple[Any, ...]:
        """
        Chống duplicate hypothesis.

        Key gồm:
            - token_ids
            - progress constraint
            - ended

        Không dùng score trong key.
        Nếu trùng key, giữ bản score cao hơn.
        """

        return (
            tuple(
                int(
                    token_id
                )
                for token_id in hyp.token_ids
            ),
            self._progress_key(
                hyp
            ),
            bool(
                hyp.ended
            ),
        )

    def _deduplicate(
        self,
        hyps: Iterable[Hypothesis],
    ) -> List[Hypothesis]:
        if not self.deduplicate:
            return list(
                hyps
            )

        best_by_key: Dict[Tuple[Any, ...], Hypothesis] = {}

        for hyp in hyps:
            key = self._dedupe_key(
                hyp
            )

            if key not in best_by_key:
                best_by_key[
                    key
                ] = hyp
                continue

            old = best_by_key[
                key
            ]

            if self._score(
                hyp
            ) > self._score(
                old
            ):
                best_by_key[
                    key
                ] = hyp

        return list(
            best_by_key.values()
        )

    # --------------------------------------------------
    # Allocation
    # --------------------------------------------------

    def _bank_sort_order(
        self,
        bank_ids: Iterable[int],
    ) -> List[int]:
        bank_ids = sorted(
            set(
                int(
                    bank_id
                )
                for bank_id in bank_ids
            )
        )

        if self.prefer_higher_banks:
            return list(
                reversed(
                    bank_ids
                )
            )

        return bank_ids

    def _allocate_slots(
        self,
        grouped: Dict[int, List[Hypothesis]],
    ) -> Dict[int, int]:
        """
        Dynamic beam allocation.

        Mỗi bank có hypothesis sẽ nhận ít nhất 1 slot,
        nếu tổng số bank <= beam_size.

        Nếu số bank > beam_size:
            ưu tiên bank cao hơn nếu prefer_higher_banks=True.

        Slot dư sẽ phân phối cho các bank còn nhiều hypothesis.
        """

        active_banks = [
            bank_id
            for bank_id, hyps in grouped.items()
            if hyps
        ]

        if not active_banks:
            return {}

        ordered_banks = self._bank_sort_order(
            active_banks
        )

        allocation: Dict[int, int] = {
            bank_id: 0
            for bank_id in active_banks
        }

        remaining = self.beam_size

        # Pass 1: mỗi bank một slot nếu còn chỗ.
        for bank_id in ordered_banks:
            if remaining <= 0:
                break

            allocation[
                bank_id
            ] += 1

            remaining -= 1

        # Pass 2: phân slot dư cho bank còn nhiều candidate.
        while remaining > 0:
            progressed = False

            for bank_id in ordered_banks:
                if remaining <= 0:
                    break

                current_slots = allocation[
                    bank_id
                ]

                available = len(
                    grouped[
                        bank_id
                    ]
                )

                if current_slots >= available:
                    continue

                allocation[
                    bank_id
                ] += 1

                remaining -= 1
                progressed = True

            if not progressed:
                break

        return allocation

    # --------------------------------------------------
    # Add / prune
    # --------------------------------------------------

    def add(
        self,
        hyp: Optional[Hypothesis],
    ) -> None:
        if hyp is None:
            return

        self.hypotheses.append(
            hyp
        )

    def extend(
        self,
        hyps: Iterable[Optional[Hypothesis]],
    ) -> None:
        if not hyps:
            return

        for hyp in hyps:
            if hyp is not None:
                self.add(
                    hyp
                )

    def prune(
        self,
    ) -> None:
        """
        Prune hypothesis bằng DBA.

        Output:
            self.hypotheses giữ tối đa beam_size hypothesis.
        """

        self.last_debug = {
            "input_count": len(
                self.hypotheses
            ),
            "deduplicated_count": None,
            "grouped_counts": {},
            "allocation": {},
            "selected_count": 0,
            "selected_banks": {},
        }

        if not self.hypotheses:
            return

        candidates = self._deduplicate(
            self.hypotheses
        )

        self.last_debug[
            "deduplicated_count"
        ] = len(
            candidates
        )

        grouped = self._group_by_bank(
            candidates
        )

        for bank_id, hyps in grouped.items():
            grouped[
                bank_id
            ] = self._sort_hyps(
                hyps
            )

        self.last_debug[
            "grouped_counts"
        ] = {
            int(
                bank_id
            ): len(
                hyps
            )
            for bank_id, hyps in grouped.items()
        }

        allocation = self._allocate_slots(
            grouped
        )

        self.last_debug[
            "allocation"
        ] = dict(
            allocation
        )

        selected: List[Hypothesis] = []

        selected_keys = set()

        # Chọn theo allocation từng bank.
        for bank_id in self._bank_sort_order(
            allocation.keys()
        ):
            slots = allocation.get(
                bank_id,
                0,
            )

            if slots <= 0:
                continue

            bank_hyps = grouped.get(
                bank_id,
                [],
            )

            for hyp in bank_hyps[
                :slots
            ]:
                key = id(
                    hyp
                )

                if key in selected_keys:
                    continue

                selected.append(
                    hyp
                )

                selected_keys.add(
                    key
                )

        # Nếu vẫn chưa đủ beam_size, fill bằng global best còn lại.
        if len(
            selected
        ) < self.beam_size:
            global_sorted = self._sort_hyps(
                candidates
            )

            for hyp in global_sorted:
                if len(
                    selected
                ) >= self.beam_size:
                    break

                key = id(
                    hyp
                )

                if key in selected_keys:
                    continue

                selected.append(
                    hyp
                )

                selected_keys.add(
                    key
                )

        selected = self._sort_hyps(
            selected
        )[
            : self.beam_size
        ]

        self.hypotheses = selected

        self.last_debug[
            "selected_count"
        ] = len(
            selected
        )

        selected_grouped = self._group_by_bank(
            selected
        )

        self.last_debug[
            "selected_banks"
        ] = {
            int(
                bank_id
            ): len(
                hyps
            )
            for bank_id, hyps in selected_grouped.items()
        }

    # --------------------------------------------------
    # Selection
    # --------------------------------------------------

    def topk(
        self,
        k: Optional[int] = None,
    ) -> List[Hypothesis]:
        k = self.beam_size if k is None else int(
            k
        )

        return self._sort_hyps(
            self.hypotheses
        )[
            :k
        ]

    def best(
        self,
        require_all_constraints_done: bool = False,
    ) -> Optional[Hypothesis]:
        if not self.hypotheses:
            return None

        candidates = list(
            self.hypotheses
        )

        if require_all_constraints_done:
            candidates = [
                hyp
                for hyp in candidates
                if hasattr(
                    hyp,
                    "all_forced_done",
                )
                and hyp.all_forced_done()
            ]

            if not candidates:
                return None

        return max(
            candidates,
            key=self._score,
        )

    def best_completed(
        self,
        require_all_constraints_done: bool = True,
    ) -> Optional[Hypothesis]:
        candidates = [
            hyp
            for hyp in self.hypotheses
            if hyp.ended
        ]

        if require_all_constraints_done:
            candidates = [
                hyp
                for hyp in candidates
                if hasattr(
                    hyp,
                    "all_forced_done",
                )
                and hyp.all_forced_done()
            ]

        if not candidates:
            return None

        return max(
            candidates,
            key=self._score,
        )

    def unfinished(
        self,
    ) -> List[Hypothesis]:
        return [
            hyp
            for hyp in self.hypotheses
            if not hyp.ended
        ]

    def completed(
        self,
    ) -> List[Hypothesis]:
        return [
            hyp
            for hyp in self.hypotheses
            if hyp.ended
        ]

    # --------------------------------------------------
    # Status
    # --------------------------------------------------

    def is_empty(
        self,
    ) -> bool:
        return len(
            self.hypotheses
        ) == 0

    def all_ended(
        self,
    ) -> bool:
        if not self.hypotheses:
            return False

        return all(
            hyp.ended
            for hyp in self.hypotheses
        )

    def clear(
        self,
    ) -> None:
        self.hypotheses = []

    def debug_info(
        self,
    ) -> Dict[str, Any]:
        return dict(
            self.last_debug
        )

    def banks_summary(
        self,
    ) -> Dict[int, int]:
        grouped = self._group_by_bank(
            self.hypotheses
        )

        return {
            int(
                bank_id
            ): len(
                hyps
            )
            for bank_id, hyps in grouped.items()
        }

    # --------------------------------------------------
    # Python protocol
    # --------------------------------------------------

    def __len__(
        self,
    ) -> int:
        return len(
            self.hypotheses
        )

    def __iter__(
        self,
    ) -> Iterator[Hypothesis]:
        return iter(
            self.hypotheses
        )

    def __getitem__(
        self,
        index: int,
    ) -> Hypothesis:
        return self.hypotheses[
            index
        ]