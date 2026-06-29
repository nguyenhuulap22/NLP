from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from terminology.constraint import (
    Constraint,
    ConstraintType,
    normalize_source_key,
    normalize_space,
    parse_bool,
    priority_score,
)
from terminology.matcher import PhraseMatcher
from terminology.glossary import GlossaryTerm


class TerminologyDetector:
    """
    Nhận diện thuật ngữ trong câu nguồn.

    Vai trò duy nhất:

        preprocess_result + glossary
        -> List[Constraint]

    Detector KHÔNG:
        - build FSA
        - đọc logits
        - đọc attention
        - quyết định active constraint
        - validate output

    Detector CÓ:
        - longest-match-wins
        - chống overlap
        - gắn source word span
        - gắn source token span
        - gắn source order
        - giữ metadata từ GlossaryTerm

    Quy ước span toàn project:

        word_span  = [start, end)
        token_span = [start, end)
        char_span  = [start, end)

    Yêu cầu:
        PhraseMatcher.match() cũng phải trả [start, end).
    """

    def __init__(
        self,
        glossary,
    ):
        self.glossary = glossary
        self.matcher = PhraseMatcher()

    # --------------------------------------------------
    # Basic helpers
    # --------------------------------------------------

    def _word_count(
        self,
        text: Any,
    ) -> int:
        text = normalize_space(
            text
        )

        if not text:
            return 0

        return len(
            text.split()
        )

    def _term_key(
        self,
        text: Any,
    ) -> str:
        return normalize_source_key(
            text
        )

    def _constraint_type_score(
        self,
        constraint_type: ConstraintType,
    ) -> int:
        return {
            ConstraintType.SOFT: 1,
            ConstraintType.HARD: 2,
            ConstraintType.PROTECTED: 3,
        }.get(
            constraint_type,
            1,
        )

    def _force_score(
        self,
        force: Any,
    ) -> int:
        return 1 if parse_bool(
            force,
            default=False,
        ) else 0

    def _protect_score(
        self,
        protect: Any,
    ) -> int:
        return 1 if parse_bool(
            protect,
            default=False,
        ) else 0

    # --------------------------------------------------
    # Preprocess result access
    # --------------------------------------------------

    def _get_words(
        self,
        preprocess_result,
    ) -> List[str]:
        words = getattr(
            preprocess_result,
            "words",
            None,
        )

        if words is None:
            raise ValueError(
                "preprocess_result thiếu field .words"
            )

        return list(
            words
        )

    def _get_alignments(
        self,
        preprocess_result,
    ) -> List[Any]:
        alignments = getattr(
            preprocess_result,
            "word_alignment",
            None,
        )

        if alignments is None:
            raise ValueError(
                "preprocess_result thiếu field .word_alignment"
            )

        return list(
            alignments
        )

    def _get_alignment_attr(
        self,
        alignment,
        name: str,
        default=None,
    ):
        if alignment is None:
            return default

        if isinstance(
            alignment,
            dict,
        ):
            return alignment.get(
                name,
                default,
            )

        return getattr(
            alignment,
            name,
            default,
        )

    # --------------------------------------------------
    # Span conversion
    # --------------------------------------------------

    def _validate_word_span(
        self,
        start: int,
        end: int,
        words: List[str],
    ) -> Optional[Tuple[int, int]]:
        """
        Chuẩn hóa matcher span theo chuẩn [start, end).

        Nếu matcher trả lỗi hoặc span ngoài biên thì bỏ.
        """

        start = int(
            start
        )

        end = int(
            end
        )

        if start < 0:
            return None

        if end <= start:
            return None

        if start >= len(
            words
        ):
            return None

        if end > len(
            words
        ):
            return None

        return start, end

    def _word_span_to_token_span(
        self,
        word_start: int,
        word_end: int,
        alignments: List[Any],
    ) -> Optional[Tuple[int, int]]:
        """
        Input:
            word_start, word_end là [start, end)

        Output:
            token_span cũng là [token_start, token_end)
        """

        if word_start < 0 or word_end <= word_start:
            return None

        if word_start >= len(
            alignments
        ):
            return None

        last_word_index = word_end - 1

        if last_word_index >= len(
            alignments
        ):
            return None

        selected = alignments[
            word_start:word_end
        ]

        token_starts = []
        token_ends = []

        for alignment in selected:
            token_start = self._get_alignment_attr(
                alignment,
                "token_start",
                None,
            )

            token_end = self._get_alignment_attr(
                alignment,
                "token_end",
                None,
            )

            if token_start is None or token_end is None:
                continue

            token_starts.append(
                int(
                    token_start
                )
            )

            token_ends.append(
                int(
                    token_end
                )
            )

        if not token_starts or not token_ends:
            return None

        token_start = min(
            token_starts
        )

        token_end = max(
            token_ends
        )

        if token_end <= token_start:
            return None

        return token_start, token_end

    def _word_span_to_char_span(
        self,
        word_start: int,
        word_end: int,
        alignments: List[Any],
    ) -> Optional[Tuple[int, int]]:
        """
        Char span là optional.

        Nếu WordAlignment không có char_start/char_end thì trả None.
        Thuật toán chính không phụ thuộc char_span.
        """

        if word_start < 0 or word_end <= word_start:
            return None

        selected = alignments[
            word_start:word_end
        ]

        char_starts = []
        char_ends = []

        for alignment in selected:
            char_start = self._get_alignment_attr(
                alignment,
                "char_start",
                None,
            )

            char_end = self._get_alignment_attr(
                alignment,
                "char_end",
                None,
            )

            if char_start is None or char_end is None:
                continue

            char_starts.append(
                int(
                    char_start
                )
            )

            char_ends.append(
                int(
                    char_end
                )
            )

        if not char_starts or not char_ends:
            return None

        char_start = min(
            char_starts
        )

        char_end = max(
            char_ends
        )

        if char_end <= char_start:
            return None

        return char_start, char_end

    # --------------------------------------------------
    # Overlap
    # --------------------------------------------------

    def _spans_overlap(
        self,
        span_a: Optional[Tuple[int, int]],
        span_b: Optional[Tuple[int, int]],
    ) -> bool:
        if span_a is None or span_b is None:
            return False

        start_a, end_a = span_a
        start_b, end_b = span_b

        return int(
            start_a
        ) < int(
            end_b
        ) and int(
            start_b
        ) < int(
            end_a
        )

    def _has_overlap_with_selected(
        self,
        constraint: Constraint,
        selected: List[Constraint],
    ) -> bool:
        for existing in selected:
            if self._spans_overlap(
                constraint.word_span,
                existing.word_span,
            ):
                return True

            if self._spans_overlap(
                constraint.token_span,
                existing.token_span,
            ):
                return True

        return False

    # --------------------------------------------------
    # Scoring / sorting
    # --------------------------------------------------

    def _term_score(
        self,
        term: GlossaryTerm,
    ):
        """
        Score dùng để sort glossary trước khi match.

        Ưu tiên:
            1. source phrase dài hơn
            2. source text dài hơn
            3. priority cao hơn
            4. protected/hard cao hơn soft
            5. force=True
            6. protect=True

        Mục tiêu:
            JSON response trước JSON
            next token trước token
            database query trước database/query
        """

        return (
            self._word_count(
                term.source
            ),
            len(
                normalize_space(
                    term.source
                )
            ),
            priority_score(
                term.priority
            ),
            self._constraint_type_score(
                term.constraint_type
            ),
            self._force_score(
                term.force
            ),
            self._protect_score(
                term.protect
            ),
        )

    def _sort_glossary_terms(
        self,
    ) -> List[GlossaryTerm]:
        terms = list(
            self.glossary.items()
        )

        return sorted(
            terms,
            key=self._term_score,
            reverse=True,
        )

    def _constraint_span_length(
        self,
        constraint: Constraint,
    ) -> int:
        if constraint.word_span is None:
            return 0

        start, end = constraint.word_span

        return max(
            0,
            int(
                end
            )
            - int(
                start
            ),
        )

    def _constraint_score(
        self,
        constraint: Constraint,
    ):
        """
        Score dùng để chọn constraint khi có overlap.

        Ưu tiên:
            1. span dài hơn
            2. source phrase dài hơn
            3. priority cao hơn
            4. protected/hard cao hơn soft
            5. force=True
            6. protect=True
            7. xuất hiện sớm hơn
        """

        start = (
            constraint.word_span[
                0
            ]
            if constraint.word_span is not None
            else 999999
        )

        return (
            self._constraint_span_length(
                constraint
            ),
            self._word_count(
                constraint.source_phrase
            ),
            len(
                constraint.source_phrase
            ),
            priority_score(
                constraint.priority
            ),
            self._constraint_type_score(
                constraint.constraint_type
            ),
            self._force_score(
                constraint.force
            ),
            self._protect_score(
                constraint.protect
            ),
            -int(
                start
            ),
        )

    def _sort_candidates_for_selection(
        self,
        candidates: List[Constraint],
    ) -> List[Constraint]:
        return sorted(
            candidates,
            key=self._constraint_score,
            reverse=True,
        )

    def _sort_selected_by_source_order(
        self,
        constraints: List[Constraint],
    ) -> List[Constraint]:
        selected = sorted(
            constraints,
            key=lambda constraint: (
                constraint.word_span[
                    0
                ]
                if constraint.word_span is not None
                else 999999,
                constraint.word_span[
                    1
                ]
                if constraint.word_span is not None
                else 999999,
                -priority_score(
                    constraint.priority
                ),
            ),
        )

        for index, constraint in enumerate(
            selected
        ):
            constraint.source_order = index

        return selected

    # --------------------------------------------------
    # Candidate build
    # --------------------------------------------------

    def _build_constraint(
        self,
        term: GlossaryTerm,
        matcher_start: int,
        matcher_end: int,
        words: List[str],
        alignments: List[Any],
    ) -> Optional[Constraint]:
        word_span = self._validate_word_span(
            matcher_start,
            matcher_end,
            words,
        )

        if word_span is None:
            return None

        word_start, word_end = word_span

        token_span = self._word_span_to_token_span(
            word_start=word_start,
            word_end=word_end,
            alignments=alignments,
        )

        if token_span is None:
            return None

        char_span = self._word_span_to_char_span(
            word_start=word_start,
            word_end=word_end,
            alignments=alignments,
        )

        matched_source_text = " ".join(
            words[
                word_start:word_end
            ]
        )

        constraint = Constraint.create(
            source_phrase=term.source,
            target_phrase=term.target,
            category=term.category,
            priority=term.priority,
            constraint_type=term.constraint_type,
            force=term.force,
            protect=term.protect,
            word_span=(
                word_start,
                word_end,
            ),
            token_span=token_span,
            char_span=char_span,
            alternatives=term.alternatives,
            meta={
                "glossary_source": term.source,
                "glossary_target": term.target,
                "matched_source_text": matched_source_text,
                "glossary_key": self._term_key(
                    term.source
                ),
                "match_length": word_end - word_start,
                "term_meta": dict(
                    getattr(
                        term,
                        "meta",
                        {},
                    )
                ),
            },
        )

        return constraint

    def _candidate_key(
        self,
        constraint: Constraint,
    ):
        return (
            constraint.word_span,
            constraint.token_span,
            self._term_key(
                constraint.source_phrase
            ),
            normalize_space(
                constraint.target_phrase
            ),
        )

    def _deduplicate_candidates(
        self,
        candidates: List[Constraint],
    ) -> List[Constraint]:
        best_by_key: Dict[Any, Constraint] = {}

        for constraint in candidates:
            key = self._candidate_key(
                constraint
            )

            if key not in best_by_key:
                best_by_key[
                    key
                ] = constraint
                continue

            old = best_by_key[
                key
            ]

            if self._constraint_score(
                constraint
            ) > self._constraint_score(
                old
            ):
                best_by_key[
                    key
                ] = constraint

        return list(
            best_by_key.values()
        )

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def detect(
        self,
        preprocess_result,
    ) -> List[Constraint]:
        """
        Detect thuật ngữ từ preprocess_result.

        preprocess_result cần có:
            .words
            .word_alignment

        word_alignment[i] cần có:
            token_start
            token_end

        Yêu cầu:
            PhraseMatcher.match(words, phrase)
            trả về List[(start, end)] với span [start, end).
        """

        words = self._get_words(
            preprocess_result
        )

        alignments = self._get_alignments(
            preprocess_result
        )

        if len(
            words
        ) != len(
            alignments
        ):
            raise ValueError(
                "preprocess_result.words và "
                "preprocess_result.word_alignment không cùng độ dài."
            )

        glossary_terms = self._sort_glossary_terms()

        candidates: List[Constraint] = []

        for term in glossary_terms:
            source_phrase = normalize_space(
                term.source
            )

            if not source_phrase:
                continue

            spans = self.matcher.match(
                words,
                source_phrase,
            )

            for matcher_start, matcher_end in spans:
                constraint = self._build_constraint(
                    term=term,
                    matcher_start=matcher_start,
                    matcher_end=matcher_end,
                    words=words,
                    alignments=alignments,
                )

                if constraint is None:
                    continue

                candidates.append(
                    constraint
                )

        candidates = self._deduplicate_candidates(
            candidates
        )

        candidates = self._sort_candidates_for_selection(
            candidates
        )

        selected: List[Constraint] = []

        for constraint in candidates:
            if self._has_overlap_with_selected(
                constraint,
                selected,
            ):
                continue

            selected.append(
                constraint
            )

        selected = self._sort_selected_by_source_order(
            selected
        )

        return selected