from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class TranslationModel:
    """
    Load model dịch EN -> VI.

    Vai trò của file này:

        - Load tokenizer
        - Load Seq2Seq model
        - Đưa model lên device
        - Ép model trả attention khi forward
        - Ép use_cache=False để manual decoding dễ lấy cross-attention

    File này KHÔNG xử lý decoding.

    Trong project này:

        - model.generate() chỉ là baseline ở Translator._normal_decode()
        - constrained decoding chính nằm ở decoding/constrained_beam_search.py
        - manual decoder loop sẽ gọi model.forward(...)
    """

    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-en-vi",
        device: Optional[str] = None,
    ):
        self.model_name = model_name

        self.device = torch.device(
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.load_error: Optional[str] = None

        self.tokenizer = self._load_tokenizer(
            model_name
        )

        self.model = self._load_model(
            model_name
        )

        self._configure_model_for_manual_decoding()

        self.model.to(
            self.device
        )

        self.model.eval()

        self.last_debug = self._build_debug_info()

    # --------------------------------------------------
    # Load tokenizer / model
    # --------------------------------------------------

    def _load_tokenizer(
        self,
        model_name: str,
    ):
        try:
            return AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
            )
        except Exception:
            return AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
            )

    def _load_model(
        self,
        model_name: str,
    ):
        """
        Ưu tiên eager attention nếu transformers hỗ trợ.

        Lý do:
            Một số backend attention tối ưu như SDPA/Flash Attention
            có thể không trả cross_attentions đầy đủ.
        """

        try:
            return AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                attn_implementation="eager",
            )

        except (
            TypeError,
            ValueError,
        ) as error:
            self.load_error = str(
                error
            )

            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
            )

            if hasattr(
                model,
                "set_attn_implementation",
            ):
                try:
                    model.set_attn_implementation(
                        "eager"
                    )
                except Exception:
                    pass

            return model

    # --------------------------------------------------
    # Config
    # --------------------------------------------------

    def _configure_model_for_manual_decoding(
        self,
    ) -> None:
        """
        Cấu hình model để phục vụ manual constrained decoding.

        Quan trọng:
            - output_attentions=True để lấy cross_attentions
            - return_dict=True để output có field rõ ràng
            - use_cache=False để mỗi step có attention đầy đủ
        """

        config = self.model.config

        config.output_attentions = True
        config.return_dict = True
        config.use_cache = False

        if hasattr(
            config,
            "_attn_implementation",
        ):
            try:
                config._attn_implementation = "eager"
            except Exception:
                pass

        if hasattr(
            config,
            "encoder",
        ):
            try:
                config.encoder.output_attentions = True
                config.encoder.return_dict = True
                config.encoder.use_cache = False

                if hasattr(
                    config.encoder,
                    "_attn_implementation",
                ):
                    config.encoder._attn_implementation = "eager"

            except Exception:
                pass

        if hasattr(
            config,
            "decoder",
        ):
            try:
                config.decoder.output_attentions = True
                config.decoder.return_dict = True
                config.decoder.use_cache = False

                if hasattr(
                    config.decoder,
                    "_attn_implementation",
                ):
                    config.decoder._attn_implementation = "eager"

            except Exception:
                pass

        self._fix_special_token_ids()

        self._configure_generation_baseline()

    def _fix_special_token_ids(
        self,
    ) -> None:
        """
        Đảm bảo các special token id có giá trị hợp lệ.

        Manual decoder cần:
            - decoder_start_token_id
            - eos_token_id
            - pad_token_id
        """

        config = self.model.config

        tokenizer_pad = getattr(
            self.tokenizer,
            "pad_token_id",
            None,
        )

        tokenizer_eos = getattr(
            self.tokenizer,
            "eos_token_id",
            None,
        )

        tokenizer_bos = getattr(
            self.tokenizer,
            "bos_token_id",
            None,
        )

        if getattr(
            config,
            "pad_token_id",
            None,
        ) is None:
            config.pad_token_id = (
                tokenizer_pad
                if tokenizer_pad is not None
                else tokenizer_eos
            )

        if getattr(
            config,
            "eos_token_id",
            None,
        ) is None:
            config.eos_token_id = tokenizer_eos

        if getattr(
            config,
            "decoder_start_token_id",
            None,
        ) is None:
            config.decoder_start_token_id = (
                tokenizer_bos
                if tokenizer_bos is not None
                else config.pad_token_id
            )

    def _configure_generation_baseline(
        self,
    ) -> None:
        """
        Cấu hình nhẹ cho baseline generate.

        Lưu ý:
            Đây không phải đường decoding chính theo bài báo.
            Đường chính là manual constrained beam search.
        """

        if not hasattr(
            self.model,
            "generation_config",
        ):
            return

        generation_config = self.model.generation_config

        generation_config.use_cache = False

        generation_config.output_attentions = False
        generation_config.return_dict_in_generate = False

        if getattr(
            generation_config,
            "decoder_start_token_id",
            None,
        ) is None:
            generation_config.decoder_start_token_id = getattr(
                self.model.config,
                "decoder_start_token_id",
                None,
            )

        if getattr(
            generation_config,
            "eos_token_id",
            None,
        ) is None:
            generation_config.eos_token_id = getattr(
                self.model.config,
                "eos_token_id",
                None,
            )

        if getattr(
            generation_config,
            "pad_token_id",
            None,
        ) is None:
            generation_config.pad_token_id = getattr(
                self.model.config,
                "pad_token_id",
                None,
            )

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def special_token_ids(
        self,
    ) -> Dict[str, Optional[int]]:
        return {
            "decoder_start_token_id": getattr(
                self.model.config,
                "decoder_start_token_id",
                None,
            ),
            "bos_token_id": getattr(
                self.model.config,
                "bos_token_id",
                None,
            ),
            "eos_token_id": getattr(
                self.model.config,
                "eos_token_id",
                None,
            ),
            "pad_token_id": getattr(
                self.model.config,
                "pad_token_id",
                None,
            ),
        }

    def _build_debug_info(
        self,
    ) -> Dict[str, Any]:
        generation_config = getattr(
            self.model,
            "generation_config",
            None,
        )

        encoder_config = getattr(
            self.model.config,
            "encoder",
            None,
        )

        decoder_config = getattr(
            self.model.config,
            "decoder",
            None,
        )

        return {
            "model_name": self.model_name,
            "model_class": self.model.__class__.__name__,
            "tokenizer_class": self.tokenizer.__class__.__name__,
            "is_fast_tokenizer": getattr(
                self.tokenizer,
                "is_fast",
                False,
            ),
            "device": str(
                self.device
            ),

            "load_error_before_fallback": self.load_error,

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

            "encoder_output_attentions": getattr(
                encoder_config,
                "output_attentions",
                None,
            ),
            "encoder_return_dict": getattr(
                encoder_config,
                "return_dict",
                None,
            ),
            "encoder_use_cache": getattr(
                encoder_config,
                "use_cache",
                None,
            ),
            "encoder_attn_implementation": getattr(
                encoder_config,
                "_attn_implementation",
                None,
            ),

            "decoder_output_attentions": getattr(
                decoder_config,
                "output_attentions",
                None,
            ),
            "decoder_return_dict": getattr(
                decoder_config,
                "return_dict",
                None,
            ),
            "decoder_use_cache": getattr(
                decoder_config,
                "use_cache",
                None,
            ),
            "decoder_attn_implementation": getattr(
                decoder_config,
                "_attn_implementation",
                None,
            ),

            "generation_output_attentions": getattr(
                generation_config,
                "output_attentions",
                None,
            )
            if generation_config is not None
            else None,

            "generation_return_dict_in_generate": getattr(
                generation_config,
                "return_dict_in_generate",
                None,
            )
            if generation_config is not None
            else None,

            "generation_use_cache": getattr(
                generation_config,
                "use_cache",
                None,
            )
            if generation_config is not None
            else None,

            "decoder_start_token_id": getattr(
                self.model.config,
                "decoder_start_token_id",
                None,
            ),
            "bos_token_id": getattr(
                self.model.config,
                "bos_token_id",
                None,
            ),
            "eos_token_id": getattr(
                self.model.config,
                "eos_token_id",
                None,
            ),
            "pad_token_id": getattr(
                self.model.config,
                "pad_token_id",
                None,
            ),
            "vocab_size": getattr(
                self.model.config,
                "vocab_size",
                None,
            ),
        }

    def debug_info(
        self,
    ) -> Dict[str, Any]:
        self.last_debug = self._build_debug_info()

        return self.last_debug