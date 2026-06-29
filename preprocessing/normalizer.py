from __future__ import annotations

import re
import unicodedata


def normalize_unicode(
    text: str,
) -> str:
    """
    Chuẩn hóa Unicode về dạng NFC.

    Vai trò:
        - Giúp tiếng Việt có dấu ổn định hơn.
        - Tránh trường hợp cùng một chữ nhưng biểu diễn Unicode khác nhau.
    """

    if text is None:
        return ""

    return unicodedata.normalize(
        "NFC",
        str(
            text
        ),
    )


def normalize_whitespace(
    text: str,
) -> str:
    """
    Chuẩn hóa khoảng trắng.

    Ví dụ:
        "hello     world" -> "hello world"

    Lưu ý:
        - Giữ newline đơn ở mức tối thiểu.
        - Gom tab / nhiều space thành một space.
        - Gom nhiều dòng trống thành một dòng trống.
    """

    if text is None:
        return ""

    text = str(
        text
    )

    # Chuẩn hóa newline Windows/Mac về Unix.
    text = text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    # Gom nhiều space/tab thành 1 space.
    text = re.sub(
        r"[ \t\f\v]+",
        " ",
        text,
    )

    # Xóa space trước newline.
    text = re.sub(
        r" *\n *",
        "\n",
        text,
    )

    # Gom quá nhiều dòng trống.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def normalize_punctuation(
    text: str,
) -> str:
    """
    Chuẩn hóa dấu câu đặc biệt về dạng phổ biến.

    Không làm:
        - Không tự thêm dấu chấm.
        - Không lowercase.
        - Không sửa thuật ngữ.
    """

    if text is None:
        return ""

    text = str(
        text
    )

    replace = {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',

        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",

        "，": ",",
        "、": ",",

        "。": ".",
        "．": ".",
        "！": "!",
        "？": "?",

        "…": "...",

        "–": "-",
        "—": "-",
        "−": "-",

        "：": ":",
        "；": ";",

        "（": "(",
        "）": ")",
        "［": "[",
        "］": "]",
        "｛": "{",
        "｝": "}",
    }

    for old, new in replace.items():
        text = text.replace(
            old,
            new,
        )

    return text


def normalize_numbers(
    text: str,
) -> str:
    """
    Chuẩn hóa số có dấu phẩy.

    Ví dụ:
        1,000      -> 1000
        1,000,000  -> 1000000

    Không đổi:
        - số thập phân kiểu 3.14
        - version đã được EntityNormalizer mask trước nếu cần
    """

    if text is None:
        return ""

    text = str(
        text
    )

    # Chỉ xóa dấu phẩy khi nó nằm giữa nhóm nghìn.
    pattern = r"\b(\d{1,3}(?:,\d{3})+)\b"

    def repl(
        match,
    ):
        return match.group(
            1
        ).replace(
            ",",
            "",
        )

    return re.sub(
        pattern,
        repl,
        text,
    )


def normalize_dates(
    text: str,
) -> str:
    """
    Chuẩn hóa ngày tháng dạng dd/mm/yyyy về yyyy-mm-dd.

    Ví dụ:
        25/06/2026 -> 2026-06-25

    Lưu ý:
        Chỉ xử lý dạng rõ ràng dd/mm/yyyy.
    """

    if text is None:
        return ""

    text = str(
        text
    )

    def repl(
        match,
    ):
        day = match.group(
            1
        )

        month = match.group(
            2
        )

        year = match.group(
            3
        )

        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

    return re.sub(
        r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b",
        repl,
        text,
    )


def normalize_case(
    text: str,
    lowercase: bool = False,
) -> str:
    """
    Chuẩn hóa chữ hoa/chữ thường.

    Mặc định không lowercase vì trong dự án IT cần giữ:
        API
        JSON
        URL
        SQL
        Transformer

    Nếu lowercase=True:
        vẫn cố giữ placeholder dạng XURL1X, XEMAIL1X...
    """

    if text is None:
        return ""

    text = str(
        text
    )

    if not lowercase:
        return text

    def repl(
        match,
    ):
        word = match.group(
            0
        )

        # Giữ placeholder entity.
        if re.fullmatch(
            r"X[A-Z]+[0-9]+X",
            word,
        ):
            return word

        # Giữ acronym IT.
        if re.fullmatch(
            r"[A-Z]{2,}",
            word,
        ):
            return word

        return word.lower()

    return re.sub(
        r"\S+",
        repl,
        text,
    )