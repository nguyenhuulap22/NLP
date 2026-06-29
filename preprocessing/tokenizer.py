from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .data import WordAlignment


class TokenizerProcessor:
    """
    Tokenizer + word-token alignment.

    Vai trò trong pipeline:

        sentence
        -> words
        -> tokenizer tokens
        -> token ids
        -> word-token alignment

    Alignment dùng chuẩn half-open span:

        token_span = [token_start, token_end)

    Ví dụ:
        word = "learning"
        token_start = 5
        token_end = 7

    Nghĩa là word này chiếm token index:

        5, 6

    File này KHÔNG:
        - detect thuật ngữ
        - build constraint
        - build FSA
        - decode
        - gọi model.generate()
    """

    def __init__(
        self,
        tokenizer,
    ):
        self.tokenizer = tokenizer

    # --------------------------------------------------
    # Word offsets
    # --------------------------------------------------

    def _find_words_with_offsets(
        self,
        sentence: str,
    ) -> List[Tuple[str, int, int]]:
        """
        Tách word bằng whitespace và lưu char span.

        Output:
            [
                ("The", 0, 3),
                ("decoder", 4, 11),
                ...
            ]

        Char span cũng dùng chuẩn:

            [char_start, char_end)
        """

        if sentence is None:
            return []

        sentence = str(
            sentence
        )

        words: List[Tuple[str, int, int]] = []
        cursor = 0

        for word in sentence.split():
            start = sentence.find(
                word,
                cursor,
            )

            if start == -1:
                continue

            end = start + len(
                word
            )

            words.append(
                (
                    word,
                    start,
                    end,
                )
            )

            cursor = end

        return words

    # --------------------------------------------------
    # Fast tokenizer path
    # --------------------------------------------------

    def _tokenize_with_offsets(
        self,
        sentence: str,
    ) -> Dict[str, Any]:
        """
        Tokenize bằng fast tokenizer.

        Cần:
            return_offsets_mapping=True

        Ưu điểm:
            Mapping word -> subword token chính xác hơn.
        """

        encoding = self.tokenizer(
            sentence,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

        ids = encoding[
            "input_ids"
        ]

        tokens = self.tokenizer.convert_ids_to_tokens(
            ids
        )

        offsets = encoding[
            "offset_mapping"
        ]

        words_with_offsets = self._find_words_with_offsets(
            sentence
        )

        alignments: List[WordAlignment] = []

        for word_index, item in enumerate(
            words_with_offsets
        ):
            word, word_start, word_end = item

            token_begin: Optional[int] = None
            token_after_end: Optional[int] = None

            for token_index, offset in enumerate(
                offsets
            ):
                token_start, token_end_offset = offset

                # Token nằm hoàn toàn trước word.
                if token_end_offset <= word_start:
                    continue

                # Token nằm hoàn toàn sau word.
                if token_start >= word_end:
                    break

                # Token giao với word span.
                if token_begin is None:
                    token_begin = token_index

                # Half-open end: token_index + 1
                token_after_end = token_index + 1

            alignments.append(
                WordAlignment(
                    word=word,
                    word_index=word_index,
                    token_start=token_begin,
                    token_end=token_after_end,
                )
            )

        return {
            "words": [
                item[
                    0
                ]
                for item in words_with_offsets
            ],
            "tokens": tokens,
            "ids": [
                int(
                    token_id
                )
                for token_id in ids
            ],
            "alignment": alignments,
        }

    # --------------------------------------------------
    # Slow tokenizer fallback
    # --------------------------------------------------

    def _encode_no_special(
        self,
        text: str,
    ) -> List[int]:
        if text is None:
            return []

        text = str(
            text
        )

        if text == "":
            return []

        return self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )

    def _tokenize_without_offsets(
        self,
        sentence: str,
    ) -> Dict[str, Any]:
        """
        Fallback cho tokenizer không hỗ trợ return_offsets_mapping.

        Ý tưởng:
            - Tokenize toàn câu để lấy ids thật.
            - Với mỗi word:
                prefix_before = text trước word
                prefix_until_word = text tới hết word
            - token_start = số token của prefix_before
            - token_end = số token của prefix_until_word

        Vì token_end là exclusive nên KHÔNG trừ 1.
        """

        ids = self._encode_no_special(
            sentence
        )

        tokens = self.tokenizer.convert_ids_to_tokens(
            ids
        )

        words_with_offsets = self._find_words_with_offsets(
            sentence
        )

        alignments: List[WordAlignment] = []

        for word_index, item in enumerate(
            words_with_offsets
        ):
            word, word_start, word_end = item

            prefix_before = sentence[
                :word_start
            ]

            prefix_until_word = sentence[
                :word_end
            ]

            token_start = len(
                self._encode_no_special(
                    prefix_before
                )
            )

            token_end = len(
                self._encode_no_special(
                    prefix_until_word
                )
            )

            if token_start >= len(
                ids
            ):
                token_start_value = None
                token_end_value = None

            elif token_end <= token_start:
                token_start_value = None
                token_end_value = None

            else:
                token_start_value = token_start
                token_end_value = min(
                    token_end,
                    len(
                        ids
                    ),
                )

            alignments.append(
                WordAlignment(
                    word=word,
                    word_index=word_index,
                    token_start=token_start_value,
                    token_end=token_end_value,
                )
            )

        return {
            "words": [
                item[
                    0
                ]
                for item in words_with_offsets
            ],
            "tokens": tokens,
            "ids": [
                int(
                    token_id
                )
                for token_id in ids
            ],
            "alignment": alignments,
        }

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def tokenize(
        self,
        sentence: str,
    ) -> Dict[str, Any]:
        """
        Tokenize một câu và trả về:

            {
                "words": [...],
                "tokens": [...],
                "ids": [...],
                "alignment": [WordAlignment(...)]
            }

        Ưu tiên fast tokenizer.
        Nếu không hỗ trợ offsets thì dùng fallback.
        """

        if sentence is None:
            sentence = ""

        sentence = str(
            sentence
        )

        try:
            return self._tokenize_with_offsets(
                sentence
            )

        except (
            NotImplementedError,
            TypeError,
            ValueError,
            KeyError,
        ):
            return self._tokenize_without_offsets(
                sentence
            )

    def debug_alignment(
        self,
        sentence: str,
    ) -> Dict[str, Any]:
        """
        Helper debug alignment.

        Dùng khi cần kiểm tra:
            word nào map vào token nào.
        """

        result = self.tokenize(
            sentence
        )

        return {
            "sentence": sentence,
            "words": result.get(
                "words",
                [],
            ),
            "tokens": result.get(
                "tokens",
                [],
            ),
            "ids": result.get(
                "ids",
                [],
            ),
            "alignment": [
                alignment.to_dict()
                if hasattr(
                    alignment,
                    "to_dict",
                )
                else alignment
                for alignment in result.get(
                    "alignment",
                    [],
                )
            ],
        }