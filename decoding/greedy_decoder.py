from __future__ import annotations

from typing import Any, Dict, List, Optional
import copy

import torch

from decoding.attention_monitor import AttentionMonitor
from constraints.constraint_activator import ConstraintActivator
from constraints.logits_masker import LogitsMasker
from constraints.covered_span_masker import CoveredSpanMasker

from terminology.constraint import Constraint


class GreedyDecoder:
    """
    Greedy Decoder debug.

    File này dùng để quan sát từng bước decoding,
    không dùng model.generate().

    Flow mỗi bước:

        model.forward(...)
        -> logits
        -> cross_attentions
        -> AttentionMonitor
        -> ConstraintActivator
        -> LogitsMasker
        -> argmax token
        -> FSA.step(token_id)
        -> mark DONE nếu FSA xong
        -> lưu trace

    File này KHÔNG:
        - dùng ConstraintManager
        - dùng ConstraintScheduler
        - rewrite/rollback output
        - inject target phrase sau khi model đã sinh sai
        - post-edit bản dịch
    """

    def __init__(
        self,
        model,
        tokenizer,
        decoder=None,
        device=None,
        attention_top_k: int = 5,
        min_focus_score: float = 0.05,
        min_span_score: float = 0.15,
        mask_eos_until_forced_done: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.decoder = decoder

        if device is None:
            try:
                device = next(
                    model.parameters()
                ).device
            except Exception:
                device = torch.device(
                    "cpu"
                )

        self.device = device

        self.attention_monitor = AttentionMonitor(
            top_k=attention_top_k,
            layer_strategy="last",
            head_strategy="mean",
            normalize=True,
        )

        self.constraint_activator = ConstraintActivator(
            min_focus_score=min_focus_score,
            min_span_score=min_span_score,
            use_topk_intersection=True,
            allow_protected=True,
        )

        self.logits_masker = LogitsMasker(
            tokenizer=self.tokenizer,
            strict=True,
            mask_eos_until_forced_done=mask_eos_until_forced_done,
        )

        self.covered_span_masker = CoveredSpanMasker(
            keep_at_least_one_token=True,
            enable_source_mask=False,
        )

        self.last_debug: Dict[str, Any] = {}

    # --------------------------------------------------
    # ID helpers
    # --------------------------------------------------

    def _decoder_start_token_id(
        self,
    ) -> int:
        candidates = [
            getattr(
                self.model.config,
                "decoder_start_token_id",
                None,
            ),
            getattr(
                getattr(
                    self.model,
                    "generation_config",
                    None,
                ),
                "decoder_start_token_id",
                None,
            ),
            getattr(
                self.tokenizer,
                "bos_token_id",
                None,
            ),
            getattr(
                self.tokenizer,
                "pad_token_id",
                None,
            ),
        ]

        for value in candidates:
            if value is not None:
                return int(
                    value
                )

        raise ValueError(
            "Không tìm thấy decoder_start_token_id / bos_token_id / pad_token_id."
        )

    def _eos_token_id(
        self,
    ) -> Optional[int]:
        candidates = [
            getattr(
                self.tokenizer,
                "eos_token_id",
                None,
            ),
            getattr(
                getattr(
                    self.model,
                    "generation_config",
                    None,
                ),
                "eos_token_id",
                None,
            ),
            getattr(
                self.model.config,
                "eos_token_id",
                None,
            ),
        ]

        for value in candidates:
            if value is not None:
                return int(
                    value
                )

        return None

    # --------------------------------------------------
    # Tensor helpers
    # --------------------------------------------------

    def _to_device_tensor(
        self,
        value,
        dtype=torch.long,
    ) -> torch.Tensor:
        if isinstance(
            value,
            torch.Tensor,
        ):
            return value.to(
                device=self.device
            )

        return torch.tensor(
            value,
            dtype=dtype,
            device=self.device,
        )

    def _prepare_attention_mask(
        self,
        attention_mask,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None

        mask = self._to_device_tensor(
            attention_mask,
            dtype=torch.long,
        )

        if mask.dim() == 1:
            mask = mask.unsqueeze(
                0
            )

        return mask

    def _build_decoder_input_ids(
        self,
        token_ids: List[int],
    ) -> torch.Tensor:
        return torch.tensor(
            [
                [
                    int(
                        token_id
                    )
                    for token_id in token_ids
                ]
            ],
            dtype=torch.long,
            device=self.device,
        )

    # --------------------------------------------------
    # Decode text helper
    # --------------------------------------------------

    def _decode_ids(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        if not token_ids:
            return ""

        try:
            return self.tokenizer.decode(
                [
                    int(
                        token_id
                    )
                    for token_id in token_ids
                ],
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=True,
            ).strip()
        except Exception:
            return ""

    def _token_from_id(
        self,
        token_id: int,
    ) -> Optional[str]:
        try:
            return self.tokenizer.convert_ids_to_tokens(
                int(
                    token_id
                )
            )
        except Exception:
            return None

    # --------------------------------------------------
    # Constraint helpers
    # --------------------------------------------------

    def _clone_constraints(
        self,
        constraints: Optional[List[Constraint]],
    ) -> List[Constraint]:
        result = []

        for constraint in constraints or []:
            if constraint is None:
                continue

            if hasattr(
                constraint,
                "clone",
            ):
                try:
                    result.append(
                        constraint.clone()
                    )
                    continue
                except Exception:
                    pass

            result.append(
                copy.deepcopy(
                    constraint
                )
            )

        return result

    def _active_constraint(
        self,
        constraints: List[Constraint],
    ) -> Optional[Constraint]:
        for constraint in constraints or []:
            if hasattr(
                constraint,
                "is_active",
            ):
                try:
                    if constraint.is_active():
                        return constraint
                except Exception:
                    pass

            state = getattr(
                constraint,
                "state",
                "",
            )

            if hasattr(
                state,
                "value",
            ):
                state = state.value

            if str(
                state
            ).upper() == "ACTIVE":
                return constraint

        return None

    def _forced_constraints(
        self,
        constraints: List[Constraint],
    ) -> List[Constraint]:
        result = []

        for constraint in constraints or []:
            if not bool(
                getattr(
                    constraint,
                    "force",
                    False,
                )
            ):
                continue

            if getattr(
                constraint,
                "fsa",
                None,
            ) is None:
                continue

            ctype = getattr(
                constraint,
                "constraint_type",
                "",
            )

            if hasattr(
                ctype,
                "value",
            ):
                ctype = ctype.value

            if str(
                ctype
            ).lower() == "soft":
                continue

            result.append(
                constraint
            )

        return result

    def _constraint_is_done(
        self,
        constraint: Constraint,
    ) -> bool:
        if hasattr(
            constraint,
            "is_done",
        ):
            try:
                if constraint.is_done():
                    return True
            except Exception:
                pass

        state = getattr(
            constraint,
            "state",
            "",
        )

        if hasattr(
            state,
            "value",
        ):
            state = state.value

        if str(
            state
        ).upper() == "DONE":
            return True

        fsa = getattr(
            constraint,
            "fsa",
            None,
        )

        if fsa is not None:
            value = getattr(
                fsa,
                "is_done",
                False,
            )

            if callable(
                value
            ):
                try:
                    return bool(
                        value()
                    )
                except Exception:
                    return False

            return bool(
                value
            )

        return False

    def _all_forced_done(
        self,
        constraints: List[Constraint],
    ) -> bool:
        forced = self._forced_constraints(
            constraints
        )

        if not forced:
            return True

        return all(
            self._constraint_is_done(
                constraint
            )
            for constraint in forced
        )

    def _forced_done_count(
        self,
        constraints: List[Constraint],
    ) -> int:
        return sum(
            1
            for constraint in self._forced_constraints(
                constraints
            )
            if self._constraint_is_done(
                constraint
            )
        )

    def _forced_total_count(
        self,
        constraints: List[Constraint],
    ) -> int:
        return len(
            self._forced_constraints(
                constraints
            )
        )

    def _sync_done_constraints(
        self,
        constraints: List[Constraint],
    ) -> None:
        for constraint in self._forced_constraints(
            constraints
        ):
            if self._constraint_is_done(
                constraint
            ):
                if hasattr(
                    constraint,
                    "mark_done",
                ):
                    constraint.mark_done()
                else:
                    constraint.state = "DONE"

    def _step_active_constraint(
        self,
        active_constraint: Optional[Constraint],
        token_id: int,
    ) -> bool:
        if active_constraint is None:
            return True

        fsa = getattr(
            active_constraint,
            "fsa",
            None,
        )

        if fsa is None:
            return True

        if not hasattr(
            fsa,
            "step",
        ):
            return True

        ok = bool(
            fsa.step(
                int(
                    token_id
                )
            )
        )

        if not ok:
            if hasattr(
                fsa,
                "fail",
            ):
                try:
                    fsa.fail()
                except Exception:
                    pass

            return False

        fsa_done = getattr(
            fsa,
            "is_done",
            False,
        )

        if callable(
            fsa_done
        ):
            try:
                fsa_done = bool(
                    fsa_done()
                )
            except Exception:
                fsa_done = False

        if bool(
            fsa_done
        ):
            if hasattr(
                active_constraint,
                "mark_done",
            ):
                active_constraint.mark_done()
            else:
                active_constraint.state = "DONE"

        return True

    def _constraints_debug(
        self,
        constraints: List[Constraint],
    ) -> List[Dict[str, Any]]:
        result = []

        for constraint in constraints or []:
            fsa = getattr(
                constraint,
                "fsa",
                None,
            )

            ctype = getattr(
                constraint,
                "constraint_type",
                None,
            )

            state = getattr(
                constraint,
                "state",
                None,
            )

            result.append(
                {
                    "id": getattr(
                        constraint,
                        "id",
                        None,
                    ),
                    "source_phrase": getattr(
                        constraint,
                        "source_phrase",
                        "",
                    ),
                    "target_phrase": getattr(
                        constraint,
                        "target_phrase",
                        "",
                    ),
                    "constraint_type": ctype.value
                    if hasattr(
                        ctype,
                        "value",
                    )
                    else str(
                        ctype
                    ),
                    "state": state.value
                    if hasattr(
                        state,
                        "value",
                    )
                    else str(
                        state
                    ),
                    "force": bool(
                        getattr(
                            constraint,
                            "force",
                            False,
                        )
                    ),
                    "covered": bool(
                        getattr(
                            constraint,
                            "covered",
                            False,
                        )
                    ),
                    "token_span": getattr(
                        constraint,
                        "token_span",
                        None,
                    ),
                    "has_fsa": fsa is not None,
                    "fsa_position": getattr(
                        fsa,
                        "position",
                        None,
                    )
                    if fsa is not None
                    else None,
                    "fsa_length": getattr(
                        fsa,
                        "length",
                        None,
                    )
                    if fsa is not None
                    else None,
                    "fsa_done": self._constraint_is_done(
                        constraint
                    ),
                }
            )

        return result

    # --------------------------------------------------
    # Model forward
    # --------------------------------------------------

    def _forward_step(
        self,
        encoder_outputs,
        attention_mask: Optional[torch.Tensor],
        decoder_input_ids: torch.Tensor,
    ):
        """
        Ưu tiên gọi decoder.step nếu wrapper tồn tại.
        Nếu không, gọi trực tiếp model(...).
        """

        if self.decoder is not None and hasattr(
            self.decoder,
            "step",
        ):
            return self.decoder.step(
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
            )

        return self.model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )

    # --------------------------------------------------
    # Decode
    # --------------------------------------------------

    def decode(
        self,
        encoder_outputs,
        attention_mask,
        constraints: Optional[List[Constraint]] = None,
        max_length: int = 128,
    ) -> Dict[str, Any]:
        constraints = self._clone_constraints(
            constraints
        )

        attention_mask = self._prepare_attention_mask(
            attention_mask
        )

        start_id = self._decoder_start_token_id()
        eos_token_id = self._eos_token_id()

        token_ids: List[int] = [
            start_id
        ]

        traces: List[Dict[str, Any]] = []

        stopped_reason = None

        for step in range(
            1,
            int(
                max_length
            )
            + 1,
        ):
            partial_before = self._decode_ids(
                token_ids[
                    1:
                ],
                skip_special_tokens=True,
            )

            hyp_attention_mask = self.covered_span_masker.apply(
                attention_mask=attention_mask,
                constraints=constraints,
            )

            decoder_input_ids = self._build_decoder_input_ids(
                token_ids
            )

            outputs = self._forward_step(
                encoder_outputs=encoder_outputs,
                attention_mask=hyp_attention_mask,
                decoder_input_ids=decoder_input_ids,
            )

            logits = outputs.logits[
                :,
                -1,
                :,
            ]

            cross_attentions = getattr(
                outputs,
                "cross_attentions",
                None,
            )

            attention_info = self.attention_monitor.get_focus(
                cross_attentions=cross_attentions,
                batch_index=0,
                target_index=-1,
                source_attention_mask=hyp_attention_mask,
            )

            activation_decision = self.constraint_activator.activate(
                constraints=constraints,
                attention_info=attention_info,
                generated_token_ids=token_ids,
                step=step,
            )

            active_constraint = self._active_constraint(
                constraints
            )

            masked_logits = self.logits_masker.apply(
                logits=logits,
                active_constraint=active_constraint,
                constraints=constraints,
                eos_token_id=eos_token_id,
            )

            masked_debug = self.logits_masker.debug_info()

            next_token_tensor = torch.argmax(
                masked_logits,
                dim=-1,
            )

            token_id = int(
                next_token_tensor.item()
            )

            token_ids.append(
                token_id
            )

            fsa_transition_ok = self._step_active_constraint(
                active_constraint=active_constraint,
                token_id=token_id,
            )

            if not fsa_transition_ok:
                stopped_reason = "fsa_transition_failed"
                break

            self._sync_done_constraints(
                constraints
            )

            self.covered_span_masker.mark_covered(
                constraints
            )

            partial_after = self._decode_ids(
                token_ids[
                    1:
                ],
                skip_special_tokens=True,
            )

            traces.append(
                {
                    "step": step,
                    "token_id": token_id,
                    "token": self._token_from_id(
                        token_id
                    ),
                    "partial_before": partial_before,
                    "partial_after": partial_after,

                    "attention": attention_info.to_dict(
                        compact=True
                    ),
                    "activation": activation_decision.to_dict(
                        compact=True
                    ),
                    "logits_masker": masked_debug,
                    "covered_span_masker": self.covered_span_masker.debug_info(),

                    "active_constraint": {
                        "id": getattr(
                            active_constraint,
                            "id",
                            None,
                        ),
                        "source_phrase": getattr(
                            active_constraint,
                            "source_phrase",
                            None,
                        ),
                        "target_phrase": getattr(
                            active_constraint,
                            "target_phrase",
                            None,
                        ),
                    }
                    if active_constraint is not None
                    else None,

                    "forced_done": self._forced_done_count(
                        constraints
                    ),
                    "forced_total": self._forced_total_count(
                        constraints
                    ),
                    "all_forced_done": self._all_forced_done(
                        constraints
                    ),
                    "constraints": self._constraints_debug(
                        constraints
                    ),
                }
            )

            if eos_token_id is not None and token_id == int(
                eos_token_id
            ):
                stopped_reason = "eos_token"
                break

        if stopped_reason is None:
            stopped_reason = "max_length_reached"

        generated_ids = token_ids[
            1:
        ]

        translation = self._decode_ids(
            generated_ids,
            skip_special_tokens=True,
        )

        self.last_debug = {
            "stopped_reason": stopped_reason,
            "steps": len(
                traces
            ),
            "forced_done": self._forced_done_count(
                constraints
            ),
            "forced_total": self._forced_total_count(
                constraints
            ),
            "all_forced_done": self._all_forced_done(
                constraints
            ),
        }

        return {
            "translation": translation,
            "generated_ids": generated_ids,
            "decoder_input_ids": self._build_decoder_input_ids(
                token_ids
            ),
            "trace": traces,
            "constraints": constraints,
            "constraints_debug": self._constraints_debug(
                constraints
            ),
            "beam_summary": {},
            "score": None,
            "debug": dict(
                self.last_debug
            ),
        }