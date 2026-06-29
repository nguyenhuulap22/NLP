from __future__ import annotations

import re
from typing import List


ABBREVIATIONS = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "st.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "fig.",
    "eq.",
    "ref.",
    "no.",
    "inc.",
    "ltd.",
    "co.",
    "corp.",
}


SUBJECT_STARTERS = {
    "i",
    "we",
    "you",
    "he",
    "she",
    "they",
    "it",
    "this",
    "that",
    "these",
    "those",
    "there",
    "the",
    "a",
    "an",
}


AUX_OR_VERB_HINTS = {
    "am",
    "is",
    "are",
    "was",
    "were",
    "will",
    "would",
    "can",
    "could",
    "should",
    "may",
    "might",
    "must",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "uses",
    "use",
    "used",
    "sends",
    "send",
    "sent",
    "receives",
    "receive",
    "received",
    "generates",
    "generate",
    "generated",
    "trains",
    "train",
    "trained",
    "deploys",
    "deploy",
    "deployed",
    "fixes",
    "fix",
    "fixed",
    "creates",
    "create",
    "created",
    "returns",
    "return",
    "returned",
}


DOMAIN_WORDS = {
    "model",
    "api",
    "server",
    "database",
    "query",
    "decoder",
    "encoder",
    "transformer",
    "attention",
    "logits",
    "token",
    "application",
    "system",
    "dataset",
    "index",
    "request",
    "response",
    "json",
    "code",
    "bug",
}


CONJUNCTIONS_OR_PREPOSITIONS = {
    "and",
    "or",
    "but",
    "because",
    "with",
    "to",
    "of",
    "in",
    "on",
    "for",
    "from",
    "by",
    "as",
    "at",
    "into",
    "than",
    "then",
}


def _clean_sentence(
    sentence: str,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        sentence,
    ).strip()


def _is_abbreviation_before_boundary(
    text: str,
    boundary_index: int,
) -> bool:
    """
    Kiểm tra dấu . tại boundary_index có thuộc abbreviation không.

    boundary_index là vị trí của dấu . ! ? trong text.
    """

    left = text[
        : boundary_index + 1
    ]

    match = re.search(
        r"([A-Za-z](?:[A-Za-z]*|(?:\.[A-Za-z])+)\.)$",
        left,
    )

    if not match:
        return False

    candidate = match.group(
        1
    ).lower()

    return candidate in ABBREVIATIONS


def split_by_punctuation(
    text: str,
) -> List[str]:
    """
    Tách câu theo . ! ? nhưng tránh tách abbreviation.

    Ví dụ:
        "Use e.g. beam search. It works."
        -> ["Use e.g. beam search.", "It works."]
    """

    if text is None:
        return []

    text = str(
        text
    ).strip()

    if not text:
        return []

    sentences: List[str] = []
    start = 0
    i = 0

    while i < len(
        text
    ):
        char = text[
            i
        ]

        if char in ".!?":
            if char == "." and _is_abbreviation_before_boundary(
                text,
                i,
            ):
                i += 1
                continue

            # Chỉ tách nếu sau dấu câu là khoảng trắng hoặc hết chuỗi.
            next_index = i + 1

            if next_index >= len(
                text
            ) or text[
                next_index
            ].isspace():
                sentence = _clean_sentence(
                    text[
                        start : i + 1
                    ]
                )

                if sentence:
                    sentences.append(
                        sentence
                    )

                start = i + 1

        i += 1

    tail = _clean_sentence(
        text[
            start:
        ]
    )

    if tail:
        sentences.append(
            tail
        )

    return sentences


def should_start_new_sentence(
    prev_word: str,
    current_word: str,
    next_word: str,
    current_length: int,
) -> bool:
    """
    Heuristic tách câu khi input không có dấu câu.

    Cực kỳ thận trọng để không phá ngữ cảnh dịch.
    """

    if current_length < 8:
        return False

    prev = prev_word.lower().strip()
    current = current_word.lower().strip()
    next_ = next_word.lower().strip()

    if prev in CONJUNCTIONS_OR_PREPOSITIONS:
        return False

    if current in CONJUNCTIONS_OR_PREPOSITIONS:
        return False

    if current in SUBJECT_STARTERS and next_ in AUX_OR_VERB_HINTS:
        return True

    if current == "the" and next_ in DOMAIN_WORDS and current_length >= 10:
        return True

    return False


def split_without_punctuation(
    text: str,
) -> List[str]:
    """
    Tách câu khi không có dấu câu.

    Lưu ý:
        Không dùng punctuation restoration thật ở đây.
        Đây chỉ là fallback heuristic nhẹ.

    Với câu ngắn hoặc câu kỹ thuật, giữ nguyên để không mất ngữ cảnh.
    """

    if text is None:
        return []

    text = _clean_sentence(
        str(
            text
        )
    )

    if not text:
        return []

    words = text.split()

    if len(
        words
    ) <= 14:
        return [
            text
        ]

    sentences: List[str] = []
    current_sentence: List[str] = []

    for i, word in enumerate(
        words
    ):
        if 0 < i < len(
            words
        ) - 1:
            if should_start_new_sentence(
                prev_word=words[
                    i - 1
                ],
                current_word=word,
                next_word=words[
                    i + 1
                ],
                current_length=len(
                    current_sentence
                ),
            ):
                sentence = _clean_sentence(
                    " ".join(
                        current_sentence
                    )
                )

                if sentence:
                    sentences.append(
                        sentence
                    )

                current_sentence = []

        current_sentence.append(
            word
        )

    if current_sentence:
        sentence = _clean_sentence(
            " ".join(
                current_sentence
            )
        )

        if sentence:
            sentences.append(
                sentence
            )

    return sentences


def split_sentences(
    text: str,
) -> List[str]:
    """
    Hàm chính để tách câu.

    Quy tắc:
        - Có . ! ? thì tách theo punctuation.
        - Không có punctuation thì dùng heuristic nhẹ.
        - Không tự thêm dấu câu.
        - Không sửa chữ hoa/thường.
        - Không thay đổi thuật ngữ.
    """

    if text is None:
        return []

    text = str(
        text
    ).strip()

    if not text:
        return []

    if re.search(
        r"[.!?]",
        text,
    ):
        return split_by_punctuation(
            text
        )

    return split_without_punctuation(
        text
    )