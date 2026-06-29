from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


class Encoder:
    """
    Wrapper cho encoder của mô hình Seq2Seq.

    Vai trò trong pipeline:

        input_ids + attention_mask
        -> encoder
        -> encoder_outputs

    File này chỉ mã hóa câu nguồn.

    File này KHÔNG:
        - decode
        - gọi model.generate()
        - xử lý FSA
        - xử lý beam search
        - xử lý constraint activation

    Constrained decoding sẽ dùng encoder_outputs này ở mỗi bước decoder.
    """

    def __init__(
        self,
        model,
    ):
        self.model = model
        self.encoder = self._get_encoder()
        self.last_debug: Optional[Dict[str, Any]] = None

    # --------------------------------------------------
    # Encoder access
    # --------------------------------------------------

    def _get_encoder(
        self,
    ):
        """
        Lấy encoder theo cách an toàn.

        Ưu tiên:
            model.get_encoder()

        Fallback:
            model.model.encoder

        Lý do:
            Một số model HuggingFace bọc encoder khác nhau.
            Dùng get_encoder() ổn định hơn gọi cứng model.model.encoder.
        """

        if hasattr(
            self.model,
            "get_encoder",
        ):
            try:
                return self.model.get_encoder()
            except Exception:
                pass

        if hasattr(
            self.model,
            "model",
        ) and hasattr(
            self.model.model,
            "encoder",
        ):
            return self.model.model.encoder

        if hasattr(
            self.model,
            "encoder",
        ):
            return self.model.encoder

        raise RuntimeError(
            "Không tìm thấy encoder trong model Seq2Seq."
        )

    # --------------------------------------------------
    # Debug
    # --------------------------------------------------

    def debug_info(
        self,
    ) -> Optional[Dict[str, Any]]:
        return self.last_debug

    def _shape(
        self,
        tensor,
    ) -> Optional[Tuple[int, ...]]:
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

    # --------------------------------------------------
    # Encode
    # --------------------------------------------------

    def encode(
        self,
        input_ids,
        attention_mask,
    ):
        """
        Chạy encoder.

        Input:
            input_ids:
                Tensor shape [batch_size, source_length]

            attention_mask:
                Tensor shape [batch_size, source_length]

        Output:
            encoder_outputs:
                thường có field last_hidden_state

        Chú ý:
            Cross-attention không xuất hiện ở đây.
            Cross-attention xuất hiện khi gọi decoder/model.forward
            với decoder_input_ids ở decoding loop.
        """

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_attentions=True,
        )

        last_hidden_state = getattr(
            encoder_outputs,
            "last_hidden_state",
            None,
        )

        attentions = getattr(
            encoder_outputs,
            "attentions",
            None,
        )

        self.last_debug = {
            "input_ids_shape": self._shape(
                input_ids
            ),
            "attention_mask_shape": self._shape(
                attention_mask
            ),
            "has_last_hidden_state": last_hidden_state is not None,
            "last_hidden_state_shape": self._shape(
                last_hidden_state
            ),
            "has_encoder_attentions": attentions is not None,
            "num_encoder_attention_layers": len(
                attentions
            )
            if attentions is not None
            else 0,
            "encoder_outputs_type": type(
                encoder_outputs
            ).__name__,
            "encoder_class": type(
                self.encoder
            ).__name__,
            "model_class": type(
                self.model
            ).__name__,
        }

        return encoder_outputs