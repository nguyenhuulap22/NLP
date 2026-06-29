from __future__ import annotations

import re
from typing import Dict, List, Tuple


ENTITY_PATTERNS: List[Tuple[str, str]] = [
    (
        "URL",
        r"https?://[^\s]+",
    ),
    (
        "EMAIL",
        r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b",
    ),
    (
        "PHONE",
        r"\b(?:\+84|0)\d{8,10}\b",
    ),
    (
        "FILEPATH",
        r"(?:[A-Za-z]:\\[^\s]+|/[A-Za-z0-9_\-./]+)",
    ),
    (
        "VERSION",
        r"\bv?\d+(?:\.\d+){1,4}\b",
    ),
    (
        "CODE_TOKEN",
        r"`[^`]+`",
    ),
]


class EntityNormalizer:
    """
    Chuẩn hóa các thực thể đặc biệt trước khi dịch.

    Vai trò trong pipeline:

        raw text
        -> entity masking
        -> preprocessing/tokenization
        -> encoder/decoder
        -> entity decode sau cùng nếu cần

    Ví dụ:
        https://example.com -> XURL1X
        abc@gmail.com       -> XEMAIL1X
        0912345678          -> XPHONE1X
        /usr/local/bin      -> XFILEPATH1X
        v1.2.3              -> XVERSION1X
        `print(x)`          -> XCODETOKEN1X

    Lý do không dùng dạng <URL_1>:
        Một số tokenizer sẽ tách <, URL, _, 1, >
        làm alignment và decoding khó kiểm soát hơn.

    File này KHÔNG:
        - dịch
        - detect terminology
        - build FSA
        - post-edit thuật ngữ
    """

    def __init__(
        self,
    ):
        self.store: Dict[str, str] = {}

        self.counter: Dict[str, int] = {
            "URL": 0,
            "EMAIL": 0,
            "PHONE": 0,
            "FILEPATH": 0,
            "VERSION": 0,
            "CODE_TOKEN": 0,
        }

    def _make_token(
        self,
        tag: str,
    ) -> str:
        self.counter[tag] += 1

        safe_tag = tag.replace(
            "_",
            "",
        )

        return f"X{safe_tag}{self.counter[tag]}X"

    def encode(
        self,
        text: str,
    ) -> str:
        if text is None:
            return ""

        text = str(
            text
        )

        for tag, pattern in ENTITY_PATTERNS:

            def repl(
                match,
                current_tag=tag,
            ):
                token = self._make_token(
                    current_tag
                )

                self.store[token] = match.group(
                    0
                )

                return token

            text = re.sub(
                pattern,
                repl,
                text,
            )

        return text

    def decode(
        self,
        text: str,
    ) -> str:
        if text is None:
            return ""

        text = str(
            text
        )

        # Thay token dài trước để tránh trường hợp token ngắn nằm trong token dài.
        for token, value in sorted(
            self.store.items(),
            key=lambda item: len(
                item[0]
            ),
            reverse=True,
        ):
            text = text.replace(
                token,
                value,
            )

        return text

    def to_dict(
        self,
    ) -> Dict[str, str]:
        return dict(
            self.store
        )

    def reset(
        self,
    ) -> None:
        self.store.clear()

        for key in self.counter:
            self.counter[key] = 0