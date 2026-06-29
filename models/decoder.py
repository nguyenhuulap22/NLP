class Decoder:
    """
    Wrapper cho decoder của mô hình Seq2Seq.

    Nhiệm vụ:
        - Nhận decoder_input_ids
        - Nhận encoder_outputs
        - Chạy model từng bước
        - Trả về logits và cross_attentions

    File này rất quan trọng cho constrained decoding theo bài báo,
    vì AttentionMonitor cần outputs.cross_attentions để biết decoder
    đang chú ý vào source span nào.
    """

    def __init__(
        self,
        model,
    ):
        self.model = model
        self.last_debug = None

    # --------------------------------------------------
    # Debug
    # --------------------------------------------------

    def debug_info(
        self,
    ):
        return self.last_debug

    def _shape(
        self,
        tensor,
    ):
        if tensor is None:
            return None

        if hasattr(
            tensor,
            "shape",
        ):
            return tuple(
                tensor.shape
            )

        return None

    def _safe_len(
        self,
        value,
    ):
        if value is None:
            return 0

        try:
            return len(
                value
            )
        except Exception:
            return 0

    def _attention_debug(
        self,
        attentions,
        name: str,
    ):
        result = {
            f"has_{name}": attentions is not None,
            f"{name}_type": type(attentions).__name__
            if attentions is not None
            else None,
            f"{name}_len": self._safe_len(
                attentions
            ),
            f"{name}_first_type": None,
            f"{name}_first_shape": None,
            f"{name}_last_type": None,
            f"{name}_last_shape": None,
        }

        if attentions is None:
            return result

        if self._safe_len(
            attentions
        ) == 0:
            return result

        try:
            first = attentions[
                0
            ]

            result[f"{name}_first_type"] = type(
                first
            ).__name__

            result[f"{name}_first_shape"] = self._shape(
                first
            )
        except Exception:
            pass

        try:
            last = attentions[
                -1
            ]

            result[f"{name}_last_type"] = type(
                last
            ).__name__

            result[f"{name}_last_shape"] = self._shape(
                last
            )
        except Exception:
            pass

        return result

    def _build_debug(
        self,
        outputs,
        decoder_input_ids,
        attention_mask,
        error=None,
    ):
        logits = getattr(
            outputs,
            "logits",
            None,
        ) if outputs is not None else None

        cross_attentions = getattr(
            outputs,
            "cross_attentions",
            None,
        ) if outputs is not None else None

        decoder_attentions = getattr(
            outputs,
            "decoder_attentions",
            None,
        ) if outputs is not None else None

        debug = {
            "error": str(
                error
            )
            if error is not None
            else None,

            "model_class": self.model.__class__.__name__,

            "decoder_input_shape": self._shape(
                decoder_input_ids
            ),

            "attention_mask_shape": self._shape(
                attention_mask
            ),

            "output_attentions_requested": True,
            "return_dict_requested": True,
            "use_cache": False,

            "has_outputs": outputs is not None,

            "has_logits": logits is not None,
            "logits_shape": self._shape(
                logits
            ),

            "config_output_attentions": getattr(
                self.model.config,
                "output_attentions",
                None,
            ),

            "config_return_dict": getattr(
                self.model.config,
                "return_dict",
                None,
            ),

            "config_use_cache": getattr(
                self.model.config,
                "use_cache",
                None,
            ),

            "attn_implementation": getattr(
                self.model.config,
                "_attn_implementation",
                None,
            ),
        }

        debug.update(
            self._attention_debug(
                cross_attentions,
                "cross_attentions",
            )
        )

        debug.update(
            self._attention_debug(
                decoder_attentions,
                "decoder_attentions",
            )
        )

        return debug

    # --------------------------------------------------
    # Main step
    # --------------------------------------------------

    def step(
        self,
        decoder_input_ids,
        encoder_outputs,
        attention_mask,
    ):
        """
        Chạy một bước decoder.

        Args:
            decoder_input_ids:
                Tensor [batch_size, current_target_len]

            encoder_outputs:
                Output từ Encoder.encode()

            attention_mask:
                Source attention mask.
                Có thể đã được CoveredSpanMasker sửa để mask span DONE.

        Return:
            outputs:
                Seq2SeqLMOutput có:
                    outputs.logits
                    outputs.cross_attentions
        """

        try:
            outputs = self.model(
                input_ids=None,
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True,
                use_cache=False,
            )

            self.last_debug = self._build_debug(
                outputs=outputs,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                error=None,
            )

            return outputs

        except TypeError as error:
            # Một số version/model không thích input_ids=None.
            # Thử lại bằng cách bỏ input_ids khỏi kwargs.
            outputs = self.model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True,
                use_cache=False,
            )

            self.last_debug = self._build_debug(
                outputs=outputs,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                error=error,
            )

            self.last_debug["fallback_used"] = "without_input_ids"

            return outputs