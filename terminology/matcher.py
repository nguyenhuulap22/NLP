from __future__ import annotations

import re
from typing import List, Tuple


class PhraseMatcher:
    """
    Match cụm thuật ngữ trong danh sách words.

    Input:
        words = ["The", "model", "uses", "machine", "learning."]
        phrase = "machine learning"

    Output:
        [(3, 5)]

    Quy ước span toàn project:

        word_span = [start, end)

    Nghĩa là:
        (3, 5) tương ứng words[3:5]
        -> ["machine", "learning."]

    File này KHÔNG:
        - build Constraint
        - build FSA
        - đọc attention
        - decode
        - validate output
    """

    def _normalize_word(
        self,
        word: str,
    ) -> str:
        if word is None:
            return ""

        word = str(
            word
        ).strip().lower()

        # Bỏ dấu câu ở đầu/cuối word:
        # "learning." -> "learning"
        # "(API)" -> "api"
        word = re.sub(
            r"^[^\w]+|[^\w]+$",
            "",
            word,
            flags=re.UNICODE,
        )

        return word

    def _normalize_phrase_words(
        self,
        phrase: str,
    ) -> List[str]:
        if phrase is None:
            return []

        phrase = str(
            phrase
        ).strip()

        result = []

        for word in phrase.split():
            normalized = self._normalize_word(
                word
            )

            if normalized:
                result.append(
                    normalized
                )

        return result

    def match(
        self,
        words: List[str],
        phrase: str,
    ) -> List[Tuple[int, int]]:
        """
        Match exact phrase theo word-level.

        Return:
            List[(start, end)]

        Trong đó:
            start inclusive
            end exclusive

        Ví dụ:
            words  = ["The", "model", "uses", "machine", "learning."]
            phrase = "machine learning"

            return [(3, 5)]
        """

        if not words:
            return []

        phrase_words = self._normalize_phrase_words(
            phrase
        )

        n = len(
            phrase_words
        )

        if n == 0:
            return []

        normalized_words = [
            self._normalize_word(
                word
            )
            for word in words
        ]

        spans: List[Tuple[int, int]] = []

        max_start = len(
            normalized_words
        ) - n

        if max_start < 0:
            return []

        for start in range(
            max_start + 1
        ):
            end = start + n

            candidate = normalized_words[
                start:end
            ]

            if candidate == phrase_words:
                spans.append(
                    (
                        start,
                        end,
                    )
                )

        return spans