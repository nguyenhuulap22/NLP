from __future__ import annotations

from .data import PreprocessResult
from .entity import EntityNormalizer
from .normalizer import (
    normalize_unicode,
    normalize_whitespace,
    normalize_punctuation,
    normalize_numbers,
    normalize_dates,
)
from .sentence import split_sentences
from .tokenizer import TokenizerProcessor


class Preprocessor:
    """
    Điều phối toàn bộ bước tiền xử lý.

    Vai trò trong pipeline Hasler constrained decoding:

        Raw text
        -> Unicode normalization
        -> Punctuation normalization
        -> Entity masking
        -> Number normalization
        -> Date normalization
        -> Whitespace normalization
        -> Sentence splitting
        -> Tokenization
        -> Word-token alignment

    Output quan trọng cho các bước sau:

        normalized_text:
            câu nguồn đã chuẩn hóa để đưa vào encoder

        sentences:
            danh sách câu sau khi tách

        words:
            danh sách word-level token để dò glossary

        tokens:
            danh sách tokenizer-level token

        token_ids:
            ids của tokenizer

        word_alignment:
            mapping word index -> token span

        entity_store:
            mapping entity placeholder -> entity gốc

    File này KHÔNG:
        - gọi model
        - gọi decoder
        - gọi generate()
        - build FSA
        - activate constraint
        - sửa bản dịch sau decoding
    """

    def __init__(
        self,
        tokenizer,
    ):
        self.tokenizer = TokenizerProcessor(
            tokenizer
        )

    def process(
        self,
        text: str,
    ) -> PreprocessResult:
        raw = text

        if text is None:
            text = ""

        text = str(
            text
        )

        # --------------------------------------------------
        # 1. Normalize cơ bản
        # --------------------------------------------------

        text = normalize_unicode(
            text
        )

        text = normalize_punctuation(
            text
        )

        text = normalize_whitespace(
            text
        )

        # --------------------------------------------------
        # 2. Mask entity
        #
        # Giữ URL, email, số điện thoại, path... khỏi bị tokenizer
        # hoặc normalizer làm hỏng.
        # --------------------------------------------------

        entity = EntityNormalizer()

        text = entity.encode(
            text
        )

        # --------------------------------------------------
        # 3. Normalize số / ngày tháng
        # --------------------------------------------------

        text = normalize_numbers(
            text
        )

        text = normalize_dates(
            text
        )

        # Normalize lại whitespace vì các bước trên có thể tạo khoảng trắng thừa.
        text = normalize_whitespace(
            text
        )

        # --------------------------------------------------
        # 4. Sentence splitting
        # --------------------------------------------------

        sentences = split_sentences(
            text
        )

        sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence and sentence.strip()
        ]

        if not sentences and text.strip():
            sentences = [
                text.strip()
            ]

        # --------------------------------------------------
        # 5. Tokenization + word-token alignment
        # --------------------------------------------------

        all_words = []
        all_tokens = []
        all_ids = []
        all_alignment = []

        token_shift = 0
        word_shift = 0

        for sentence in sentences:
            result = self.tokenizer.tokenize(
                sentence
            )

            words = result.get(
                "words",
                [],
            )

            tokens = result.get(
                "tokens",
                [],
            )

            ids = result.get(
                "ids",
                [],
            )

            alignments = result.get(
                "alignment",
                [],
            )

            for alignment in alignments:
                alignment.word_index += word_shift

                if alignment.token_start is not None:
                    alignment.token_start += token_shift

                if alignment.token_end is not None:
                    alignment.token_end += token_shift

                all_alignment.append(
                    alignment
                )

            all_words.extend(
                words
            )

            all_tokens.extend(
                tokens
            )

            all_ids.extend(
                ids
            )

            word_shift += len(
                words
            )

            token_shift += len(
                tokens
            )

        return PreprocessResult(
            raw_text=raw,
            normalized_text=text,
            sentences=sentences,
            words=all_words,
            tokens=all_tokens,
            token_ids=all_ids,
            word_alignment=all_alignment,
            entity_store=entity.store,
        )