class SourceSpanAligner:
    """
    Gắn vị trí token nguồn cho mỗi constraint.

    Quy ước:
        token_span = [start, end)

    Ví dụ:
        token_span = [3, 5)
        encoder_positions = [3, 4]
    """

    def align(
        self,
        preprocess_result,
        constraints,
    ):
        for constraint in constraints or []:
            token_span = getattr(
                constraint,
                "token_span",
                None,
            )

            if token_span is None:
                continue

            try:
                start, end = token_span
                start = int(start)
                end = int(end)
            except Exception:
                continue

            if end <= start:
                constraint.encoder_positions = []
                continue

            constraint.encoder_positions = list(
                range(start, end)
            )

        return constraints