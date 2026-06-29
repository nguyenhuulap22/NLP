from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class WordAlignment:
    """
    Mapping giữa một từ trong câu nguồn và các subword token tương ứng.

    Ví dụ:
        word = "learning"
        word_index = 4
        token_start = 5
        token_end = 6

    Quy ước:
        token_start: inclusive
        token_end: exclusive

    Nghĩa là:
        token span [token_start, token_end)
    """

    word: str
    word_index: int
    token_start: Optional[int]
    token_end: Optional[int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "word": self.word,
            "word_index": self.word_index,
            "token_start": self.token_start,
            "token_end": self.token_end,
        }


@dataclass
class PreprocessResult:
    """
    Kết quả sau toàn bộ bước tiền xử lý.

    Đây là object trung gian dùng cho:

        Preprocessor
        -> TerminologyDetector
        -> FSABuilder
        -> Encoder
        -> Constrained Decoder

    Các field quan trọng:

        normalized_text:
            văn bản đã chuẩn hóa để đưa vào encoder

        words:
            danh sách từ ở word-level để detect thuật ngữ

        tokens:
            danh sách subword/token của tokenizer

        token_ids:
            danh sách id tương ứng với tokens

        word_alignment:
            mapping word index -> token span

        entity_store:
            mapping placeholder -> entity gốc
    """

    raw_text: str
    normalized_text: str
    sentences: List[str]
    words: List[str]
    tokens: List[str]
    token_ids: List[int]
    word_alignment: List[WordAlignment]
    entity_store: Dict[str, str] = field(default_factory=dict)

    def word_count(self) -> int:
        return len(
            self.words
        )

    def token_count(self) -> int:
        return len(
            self.tokens
        )

    def get_word(
        self,
        index: int,
    ) -> Optional[str]:
        if index < 0 or index >= len(
            self.words
        ):
            return None

        return self.words[
            index
        ]

    def get_alignment_by_word_index(
        self,
        word_index: int,
    ) -> Optional[WordAlignment]:
        for alignment in self.word_alignment:
            if alignment.word_index == word_index:
                return alignment

        return None

    def word_span_to_text(
        self,
        word_span: Tuple[int, int],
    ) -> str:
        """
        Convert word span [start, end) thành text.

        Ví dụ:
            words = ["The", "model", "uses", "machine", "learning"]
            word_span = (3, 5)

        Output:
            "machine learning"
        """

        start, end = word_span

        start = max(
            0,
            start,
        )

        end = min(
            len(
                self.words
            ),
            end,
        )

        if start >= end:
            return ""

        return " ".join(
            self.words[
                start:end
            ]
        )

    def word_span_to_token_span(
        self,
        word_span: Tuple[int, int],
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Convert word span [word_start, word_end)
        sang token span [token_start, token_end).

        Đây là bước rất quan trọng cho constrained decoding.

        Ví dụ:
            source phrase:
                machine learning

            word_span:
                [3, 5)

            token_span:
                [5, 8)

        AttentionMonitor sẽ dùng token_span/source span để biết
        decoder đang nhìn vào vùng thuật ngữ nào.
        """

        word_start, word_end = word_span

        if word_start < 0:
            word_start = 0

        if word_end > len(
            self.words
        ):
            word_end = len(
                self.words
            )

        if word_start >= word_end:
            return None, None

        selected = [
            alignment
            for alignment in self.word_alignment
            if word_start <= alignment.word_index < word_end
            and alignment.token_start is not None
            and alignment.token_end is not None
        ]

        if not selected:
            return None, None

        token_start = min(
            alignment.token_start
            for alignment in selected
            if alignment.token_start is not None
        )

        token_end = max(
            alignment.token_end
            for alignment in selected
            if alignment.token_end is not None
        )

        return token_start, token_end

    def to_dict(self) -> Dict[str, object]:
        return {
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "sentences": self.sentences,
            "words": self.words,
            "tokens": self.tokens,
            "token_ids": self.token_ids,
            "word_alignment": [
                alignment.to_dict()
                for alignment in self.word_alignment
            ],
            "entity_store": self.entity_store,
            "word_count": self.word_count(),
            "token_count": self.token_count(),
        }