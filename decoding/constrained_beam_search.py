from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import copy
import math

import torch
import torch.nn.functional as F

from decoding.hypothesis import Hypothesis
from decoding.dba_beam import DBABeam
from decoding.attention_monitor import AttentionMonitor, AttentionInfo

from constraints.constraint_activator import ConstraintActivator
from constraints.logits_masker import LogitsMasker
from constraints.covered_span_masker import CoveredSpanMasker

from terminology.constraint import Constraint


class ConstrainedBeamSearch:
    """
    Attention-guided constrained beam search.

    Có:
        - delayed forced FSA
        - hard mask khi FSA ACTIVE
        - delay branch trước khi ép FSA
        - chống EOS quá sớm
        - partial prefix sync để tránh "máy máy chủ"
        - CoveredSpanMasker để che source span đã dịch

    Không dùng:
        - model.generate()
        - force_words_ids
        - post-edit / replace sau dịch
    """

    def __init__(
        self,
        model,
        tokenizer,
        decoder=None,
        beam_size: int = 8,
        top_k: int = 20,
        max_length: int = 128,
        length_penalty: float = 1.0,
        constraint_bonus: float = 0.0,
        device=None,
        attention_top_k: int = 5,
        min_focus_score: float = 0.10,
        min_span_score: float = 0.25,
        mask_eos_until_forced_done: bool = False,
        return_trace: bool = True,
        min_decode_length: int = 4,
        min_length_ratio: float = 0.85,
        short_output_penalty: float = 3.0,
        max_delay_before_constraint: int = 2,
        delay_branch_top_k: int = 8,
        delay_penalty: float = 0.10,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.decoder = decoder

        self.beam_size = max(1, int(beam_size))
        self.top_k = max(1, int(top_k))
        self.max_length = max(1, int(max_length))
        self.length_penalty = float(length_penalty)
        self.constraint_bonus = float(constraint_bonus)

        self.min_decode_length = max(1, int(min_decode_length))
        self.min_length_ratio = max(0.0, float(min_length_ratio))
        self.short_output_penalty = max(0.0, float(short_output_penalty))

        self.max_delay_before_constraint = max(0, int(max_delay_before_constraint))
        self.delay_branch_top_k = max(1, int(delay_branch_top_k))
        self.delay_penalty = max(0.0, float(delay_penalty))

        if device is None:
            try:
                device = next(model.parameters()).device
            except Exception:
                device = torch.device("cpu")

        self.device = device
        self.return_trace = bool(return_trace)

        self._current_source_len: Optional[int] = None
        self._current_min_decode_length: int = self.min_decode_length
        self.last_debug: Dict[str, Any] = {}

        self.attention_monitor = AttentionMonitor(
            top_k=attention_top_k,
            layer_strategy="last",
            head_strategy="mean",
            normalize=True,
        )

        self.constraint_activator = ConstraintActivator(
            min_focus_score=min_focus_score,
            min_span_score=min_span_score,
            use_topk_intersection=False,
            allow_protected=True,
        )

        self.logits_masker = LogitsMasker(
            tokenizer=self.tokenizer,
            strict=True,
            mask_eos_until_forced_done=mask_eos_until_forced_done,
        )

        try:
            self.covered_span_masker = CoveredSpanMasker(
                keep_at_least_one_token=True,
                enable_source_mask=True,
            )
        except TypeError:
            self.covered_span_masker = CoveredSpanMasker(
                keep_at_least_one_token=True,
            )

    # --------------------------------------------------
    # Basic IDs
    # --------------------------------------------------

    def _decoder_start_token_id(self) -> int:
        values = [
            getattr(self.model.config, "decoder_start_token_id", None),
            getattr(
                getattr(self.model, "generation_config", None),
                "decoder_start_token_id",
                None,
            ),
            getattr(self.tokenizer, "bos_token_id", None),
            getattr(self.tokenizer, "pad_token_id", None),
        ]

        for value in values:
            if value is not None:
                return int(value)

        raise ValueError("Không tìm thấy decoder_start_token_id.")

    def _eos_token_id(self) -> Optional[int]:
        values = [
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(
                getattr(self.model, "generation_config", None),
                "eos_token_id",
                None,
            ),
            getattr(self.model.config, "eos_token_id", None),
        ]

        for value in values:
            if value is not None:
                return int(value)

        return None

    # --------------------------------------------------
    # Tensor helpers
    # --------------------------------------------------

    def _to_tensor(self, value, dtype=torch.long) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=self.device)

        return torch.tensor(
            value,
            dtype=dtype,
            device=self.device,
        )

    def _prepare_attention_mask(self, attention_mask) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None

        mask = self._to_tensor(attention_mask, dtype=torch.long)

        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        return mask

    def _decoder_input_tensor(self, hyp: Hypothesis) -> torch.Tensor:
        return torch.tensor(
            [hyp.decoder_input_ids()],
            dtype=torch.long,
            device=self.device,
        )

    # --------------------------------------------------
    # Length / EOS
    # --------------------------------------------------

    def _source_len(self, attention_mask: Optional[torch.Tensor]) -> int:
        if attention_mask is None:
            return 0

        try:
            mask = attention_mask[0] if attention_mask.dim() == 2 else attention_mask
            return int(mask.detach().long().sum().item())
        except Exception:
            return 0

    def _compute_min_decode_length(self, source_len: int) -> int:
        if source_len <= 0:
            return self.min_decode_length

        return max(
            self.min_decode_length,
            int(math.ceil(float(source_len) * self.min_length_ratio)),
        )

    def _generated_len(self, hyp: Optional[Hypothesis]) -> int:
        if hyp is None:
            return 0

        return max(0, len(getattr(hyp, "token_ids", [])) - 1)

    def _should_block_eos(
        self,
        hyp: Hypothesis,
        token_id: int,
        eos_token_id: Optional[int],
    ) -> bool:
        if eos_token_id is None:
            return False

        if int(token_id) != int(eos_token_id):
            return False

        return self._generated_len(hyp) < self._current_min_decode_length

    def _filter_eos(
        self,
        hyp: Hypothesis,
        candidates: List[Tuple[int, float]],
        eos_token_id: Optional[int],
    ) -> List[Tuple[int, float]]:
        if eos_token_id is None:
            return candidates

        return [
            (token_id, log_prob)
            for token_id, log_prob in candidates
            if not self._should_block_eos(hyp, token_id, eos_token_id)
        ]

    def _short_penalty(self, hyp: Hypothesis) -> float:
        missing = self._current_min_decode_length - self._generated_len(hyp)

        if missing <= 0:
            return 0.0

        return -float(missing) * self.short_output_penalty

    def _selection_score(self, hyp: Hypothesis) -> float:
        try:
            score = hyp.final_score(
                length_penalty=self.length_penalty,
                constraint_bonus=self.constraint_bonus,
            )
        except Exception:
            score = float(getattr(hyp, "score", -1e9))

        return float(score) + self._short_penalty(hyp)

    # --------------------------------------------------
    # Constraint helpers
    # --------------------------------------------------

    def _constraint_key(self, constraint: Optional[Constraint]) -> str:
        if constraint is None:
            return ""

        cid = getattr(constraint, "id", None)

        if cid:
            return str(cid)

        return (
            f"{getattr(constraint, 'source_phrase', '')}::"
            f"{getattr(constraint, 'target_phrase', '')}::"
            f"{getattr(constraint, 'token_span', None)}"
        )

    def _constraint_meta(self, constraint: Optional[Constraint]) -> Dict[str, Any]:
        if constraint is None:
            return {}

        meta = getattr(constraint, "meta", None)

        if isinstance(meta, dict):
            return meta

        try:
            constraint.meta = {}
            return constraint.meta
        except Exception:
            return {}

    def _delay_key(self, constraint: Optional[Constraint]) -> str:
        return f"_delay_count::{self._constraint_key(constraint)}"

    def _get_delay_count(self, constraint: Optional[Constraint]) -> int:
        if constraint is None:
            return 0

        meta = self._constraint_meta(constraint)
        key = self._delay_key(constraint)

        try:
            return int(meta.get(key, getattr(constraint, "_delay_count", 0)))
        except Exception:
            return 0

    def _set_delay_count(self, constraint: Optional[Constraint], value: int) -> None:
        if constraint is None:
            return

        value = max(0, int(value))
        meta = self._constraint_meta(constraint)
        key = self._delay_key(constraint)

        try:
            meta[key] = value
        except Exception:
            pass

        try:
            setattr(constraint, "_delay_count", value)
        except Exception:
            pass

    def _increment_delay_count(self, constraint: Optional[Constraint]) -> int:
        value = self._get_delay_count(constraint) + 1
        self._set_delay_count(constraint, value)
        return value

    def _is_constraint_done(self, constraint: Optional[Constraint]) -> bool:
        if constraint is None:
            return False

        if hasattr(constraint, "is_done"):
            try:
                return bool(constraint.is_done())
            except Exception:
                pass

        state = getattr(constraint, "state", "")
        state = getattr(state, "value", state)

        return str(state).lower() == "done"

    def _find_matching_constraint(
        self,
        constraints: List[Constraint],
        target_constraint: Optional[Constraint],
    ) -> Optional[Constraint]:
        key = self._constraint_key(target_constraint)

        for constraint in constraints or []:
            if self._constraint_key(constraint) == key:
                return constraint

        return None

    def _can_delay_constraint(self, constraint: Optional[Constraint]) -> bool:
        if constraint is None:
            return False

        if self.max_delay_before_constraint <= 0:
            return False

        if self._is_constraint_done(constraint):
            return False

        fsa = getattr(constraint, "fsa", None)

        if fsa is None:
            return False

        if int(getattr(fsa, "position", 0) or 0) > 0:
            return False

        return self._get_delay_count(constraint) < self.max_delay_before_constraint

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    def _forward_step(
        self,
        encoder_outputs,
        attention_mask: Optional[torch.Tensor],
        decoder_input_ids: torch.Tensor,
    ):
        return self.model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )

    def encode_source(self, input_ids, attention_mask=None):
        input_ids = self._to_tensor(input_ids, dtype=torch.long)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        attention_mask = self._prepare_attention_mask(attention_mask)

        if attention_mask is None:
            attention_mask = torch.ones_like(
                input_ids,
                dtype=torch.long,
                device=self.device,
            )

        encoder_outputs = self.model.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        return encoder_outputs, attention_mask

    # --------------------------------------------------
    # Decode / clone helpers
    # --------------------------------------------------

    def _decode_ids(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        try:
            return self.tokenizer.decode(
                [int(x) for x in token_ids],
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=True,
            ).strip()
        except Exception:
            return ""

    def _token_from_id(self, token_id: int) -> Optional[str]:
        try:
            return self.tokenizer.convert_ids_to_tokens(int(token_id))
        except Exception:
            return None

    def _clone_constraints(
        self,
        constraints: Optional[List[Constraint]],
    ) -> List[Constraint]:
        result = []

        for constraint in constraints or []:
            if constraint is None:
                continue

            if hasattr(constraint, "clone"):
                try:
                    result.append(constraint.clone())
                    continue
                except Exception:
                    pass

            result.append(copy.deepcopy(constraint))

        return result

    # --------------------------------------------------
    # Constraint progress
    # --------------------------------------------------

    def _forced_done_count(self, hyp: Hypothesis) -> int:
        try:
            return int(hyp.forced_done_count())
        except Exception:
            return 0

    def _forced_total_count(self, hyp: Hypothesis) -> int:
        try:
            return int(hyp.forced_total_count())
        except Exception:
            return 0

    def _all_forced_done(self, hyp: Hypothesis) -> bool:
        try:
            return bool(hyp.all_forced_done())
        except Exception:
            return True

    def _fsa_progress_sum(self, hyp: Hypothesis) -> int:
        total = 0

        for constraint in getattr(hyp, "constraints", []) or []:
            fsa = getattr(constraint, "fsa", None)

            if fsa is not None:
                total += int(getattr(fsa, "position", 0) or 0)

        return total

    def _fsa_is_done(self, fsa) -> bool:
        if fsa is None:
            return False

        value = getattr(fsa, "is_done", False)

        if callable(value):
            try:
                return bool(value())
            except Exception:
                return False

        return bool(value)

    # --------------------------------------------------
    # FSA sync
    # --------------------------------------------------

    def _mark_constraint_done(self, constraint: Constraint) -> None:
        if hasattr(constraint, "mark_done"):
            try:
                constraint.mark_done()
            except Exception:
                pass
        else:
            try:
                constraint.state = "DONE"
            except Exception:
                pass

        try:
            constraint.covered = True
        except Exception:
            pass

        self._set_delay_count(constraint, 0)

    def _sync_fsa_partial_prefix_with_generated_tail(
        self,
        constraint: Optional[Constraint],
        generated_token_ids: Sequence[int],
    ) -> None:
        """
        Nếu output đã sinh một phần đầu target phrase thì FSA nhảy tiếp.

        Ví dụ:
            generated tail: "... máy"
            target: "máy chủ"

        FSA.position = 1 để ép tiếp "chủ", tránh "máy máy chủ".
        """

        if constraint is None:
            return

        fsa = getattr(constraint, "fsa", None)

        if fsa is None:
            return

        target_ids = getattr(constraint, "target_token_ids", None)

        if not target_ids:
            target_ids = getattr(fsa, "target_token_ids", None)

        if not target_ids or not generated_token_ids:
            return

        try:
            target_ids = [int(x) for x in target_ids]
            generated_ids = [int(x) for x in generated_token_ids]
        except Exception:
            return

        current_pos = int(getattr(fsa, "position", 0) or 0)
        best_pos = current_pos
        max_len = min(len(target_ids), len(generated_ids))

        for n in range(1, max_len + 1):
            if generated_ids[-n:] == target_ids[:n]:
                best_pos = max(best_pos, n)

        if best_pos <= current_pos:
            return

        try:
            fsa.position = best_pos
        except Exception:
            return

        if best_pos >= len(target_ids):
            self._mark_constraint_done(constraint)

    def _sync_constraints_with_generated_tail(self, hyp: Hypothesis) -> None:
        if hyp is None:
            return

        token_ids = getattr(hyp, "token_ids", [])

        for constraint in getattr(hyp, "constraints", []) or []:
            if constraint is None:
                continue

            fsa = getattr(constraint, "fsa", None)

            if fsa is None:
                continue

            if hasattr(fsa, "sync_with_generated_tail"):
                try:
                    fsa.sync_with_generated_tail(token_ids)
                except Exception:
                    pass

            if self._fsa_is_done(fsa):
                self._mark_constraint_done(constraint)

    # --------------------------------------------------
    # Trace
    # --------------------------------------------------

    def _constraint_trace(
        self,
        constraint: Optional[Constraint],
    ) -> Optional[Dict[str, Any]]:
        if constraint is None:
            return None

        fsa = getattr(constraint, "fsa", None)

        return {
            "id": getattr(constraint, "id", None),
            "source": getattr(constraint, "source_phrase", None),
            "target": getattr(constraint, "target_phrase", None),
            "state": str(getattr(constraint, "state", None)),
            "covered": bool(getattr(constraint, "covered", False)),
            "fsa_position": int(getattr(fsa, "position", 0)) if fsa is not None else None,
            "delay_count": self._get_delay_count(constraint),
        }

    def _make_trace_item(
        self,
        step: int,
        hyp: Hypothesis,
        token_id: int,
        log_prob: float,
        attention_info: Optional[AttentionInfo],
        activation_decision,
        masked_debug: Dict[str, Any],
        new_hyp: Hypothesis,
        branch: str,
    ) -> Dict[str, Any]:
        active = hyp.active_constraint()
        after = new_hyp.active_constraint()

        return {
            "step": int(step),
            "branch": branch,
            "token_id": int(token_id),
            "token": self._token_from_id(token_id),
            "log_prob": float(log_prob),
            "score": float(new_hyp.score),
            "partial_before": self._decode_ids(hyp.token_ids[1:]),
            "partial_after": self._decode_ids(new_hyp.token_ids[1:]),
            "attention": attention_info.to_dict(compact=True)
            if attention_info is not None
            else None,
            "activation": activation_decision.to_dict(compact=True)
            if activation_decision is not None
            else None,
            "logits_masker": dict(masked_debug or {}),
            "active_before": self._constraint_trace(active),
            "active_after": self._constraint_trace(after),
            "forced_done": new_hyp.forced_done_count(),
            "forced_total": new_hyp.forced_total_count(),
            "bank_id": new_hyp.bank_id(),
            "constraint_progress_key": new_hyp.constraint_progress_key(),
            "generated_len": self._generated_len(new_hyp),
            "min_decode_length": self._current_min_decode_length,
            "covered_span_masker": self.covered_span_masker.debug_info()
            if hasattr(self.covered_span_masker, "debug_info")
            else {},
        }

    # --------------------------------------------------
    # Candidate helpers
    # --------------------------------------------------

    def _topk_from_logits(
        self,
        logits: torch.Tensor,
        k: int,
    ) -> List[Tuple[int, float]]:
        log_probs = F.log_softmax(logits, dim=-1)
        k = min(int(k), int(log_probs.size(-1)))

        values, indices = torch.topk(
            log_probs,
            k=k,
            dim=-1,
        )

        result = []

        for i in range(k):
            token_id = int(indices[0, i].item())
            log_prob = float(values[0, i].item())

            if math.isfinite(log_prob):
                result.append((token_id, log_prob))

        return result

    def _dedupe_hypotheses(
        self,
        hypotheses: List[Hypothesis],
    ) -> List[Hypothesis]:
        best: Dict[Tuple[Any, ...], Hypothesis] = {}

        for hyp in hypotheses or []:
            if hyp is None:
                continue

            try:
                progress_key = hyp.constraint_progress_key()
            except Exception:
                progress_key = hyp.bank_id()

            key = (
                tuple(int(x) for x in hyp.token_ids),
                progress_key,
                bool(hyp.ended),
            )

            if key not in best or self._selection_score(hyp) > self._selection_score(best[key]):
                best[key] = hyp

        return list(best.values())

    def _mark_covered_safe(
        self,
        constraints: List[Constraint],
    ) -> None:
        if hasattr(self.covered_span_masker, "mark_covered"):
            try:
                self.covered_span_masker.mark_covered(constraints)
            except Exception:
                pass

    # --------------------------------------------------
    # Expansion
    # --------------------------------------------------

    def _expand_with_forced_logits(
        self,
        base_hyp: Hypothesis,
        logits: torch.Tensor,
        attention_info: Optional[AttentionInfo],
        activation_decision,
        eos_token_id: Optional[int],
        step: int,
        branch: str,
    ) -> List[Hypothesis]:
        active = base_hyp.active_constraint()

        self._sync_fsa_partial_prefix_with_generated_tail(
            constraint=active,
            generated_token_ids=base_hyp.token_ids,
        )

        active = base_hyp.active_constraint()

        if active is None:
            return []

        masked_logits = self.logits_masker.apply(
            logits=logits,
            active_constraint=active,
            constraints=base_hyp.constraints,
            eos_token_id=eos_token_id,
        )

        masked_debug = self.logits_masker.debug_info()

        candidates = self._filter_eos(
            hyp=base_hyp,
            candidates=self._topk_from_logits(masked_logits, self.top_k),
            eos_token_id=eos_token_id,
        )

        return self._build_hypotheses(
            base_hyp=base_hyp,
            candidates=candidates,
            eos_token_id=eos_token_id,
            attention_info=attention_info,
            activation_decision=activation_decision,
            masked_debug=masked_debug,
            step=step,
            branch=branch,
        )

    def _expand_with_normal_logits(
        self,
        base_hyp: Hypothesis,
        logits: torch.Tensor,
        attention_info: Optional[AttentionInfo],
        activation_decision,
        eos_token_id: Optional[int],
        step: int,
        branch: str,
        top_k: Optional[int] = None,
        extra_log_penalty: float = 0.0,
    ) -> List[Hypothesis]:
        normal_logits = self.logits_masker.apply(
            logits=logits,
            active_constraint=None,
            constraints=base_hyp.constraints,
            eos_token_id=eos_token_id,
        )

        masked_debug = self.logits_masker.debug_info()
        k = int(top_k if top_k is not None else self.top_k)

        candidates = self._topk_from_logits(normal_logits, k + 10)
        candidates = self._filter_eos(base_hyp, candidates, eos_token_id)
        candidates = candidates[:k]

        if extra_log_penalty:
            candidates = [
                (token_id, float(log_prob) - float(extra_log_penalty))
                for token_id, log_prob in candidates
            ]

        return self._build_hypotheses(
            base_hyp=base_hyp,
            candidates=candidates,
            eos_token_id=eos_token_id,
            attention_info=attention_info,
            activation_decision=activation_decision,
            masked_debug=masked_debug,
            step=step,
            branch=branch,
        )

    def _build_hypotheses(
        self,
        base_hyp: Hypothesis,
        candidates: List[Tuple[int, float]],
        eos_token_id: Optional[int],
        attention_info: Optional[AttentionInfo],
        activation_decision,
        masked_debug: Dict[str, Any],
        step: int,
        branch: str,
    ) -> List[Hypothesis]:
        new_hyps = []

        for token_id, log_prob in candidates:
            new_hyp = base_hyp.extend(
                token_id=token_id,
                log_prob=log_prob,
                eos_token_id=eos_token_id,
                trace_item=None,
                step_constraints=True,
            )

            if new_hyp is None:
                continue

            self._sync_constraints_with_generated_tail(new_hyp)
            self._mark_covered_safe(new_hyp.constraints)

            if self.return_trace:
                new_hyp.append_trace(
                    self._make_trace_item(
                        step=step,
                        hyp=base_hyp,
                        token_id=token_id,
                        log_prob=log_prob,
                        attention_info=attention_info,
                        activation_decision=activation_decision,
                        masked_debug=masked_debug,
                        new_hyp=new_hyp,
                        branch=branch,
                    )
                )

            new_hyps.append(new_hyp)

        return self._dedupe_hypotheses(new_hyps)

    def _expand_one_hypothesis(
        self,
        hyp: Hypothesis,
        encoder_outputs,
        base_attention_mask: Optional[torch.Tensor],
        step: int,
    ) -> List[Hypothesis]:
        if hyp.ended:
            return [hyp]

        eos_token_id = self._eos_token_id()
        base_hyp = hyp.clone()

        decoder_input_ids = self._decoder_input_tensor(base_hyp)

        hyp_attention_mask = self.covered_span_masker.apply(
            attention_mask=base_attention_mask,
            constraints=base_hyp.constraints,
        )

        outputs = self._forward_step(
            encoder_outputs=encoder_outputs,
            attention_mask=hyp_attention_mask,
            decoder_input_ids=decoder_input_ids,
        )

        logits = outputs.logits[:, -1, :]
        cross_attentions = getattr(outputs, "cross_attentions", None)

        attention_info = self.attention_monitor.get_focus(
            cross_attentions=cross_attentions,
            batch_index=0,
            target_index=-1,
            source_attention_mask=hyp_attention_mask,
        )

        active = base_hyp.active_constraint()

        if active is not None:
            activation_decision = self.constraint_activator.activate(
                constraints=base_hyp.constraints,
                attention_info=attention_info,
                generated_token_ids=base_hyp.token_ids,
                step=step,
            )

            if base_hyp.active_constraint() is not None:
                return self._expand_with_forced_logits(
                    base_hyp=base_hyp,
                    logits=logits,
                    attention_info=attention_info,
                    activation_decision=activation_decision,
                    eos_token_id=eos_token_id,
                    step=step,
                    branch="continue_active_fsa_forced",
                )

        forced_base = base_hyp.clone()

        activation_decision = self.constraint_activator.activate(
            constraints=forced_base.constraints,
            attention_info=attention_info,
            generated_token_ids=forced_base.token_ids,
            step=step,
        )

        forced_active = forced_base.active_constraint()

        self._sync_fsa_partial_prefix_with_generated_tail(
            constraint=forced_active,
            generated_token_ids=forced_base.token_ids,
        )

        forced_active = forced_base.active_constraint()

        if (
            activation_decision is not None
            and activation_decision.activated
            and forced_active is not None
        ):
            result = []

            result.extend(
                self._expand_with_forced_logits(
                    base_hyp=forced_base,
                    logits=logits,
                    attention_info=attention_info,
                    activation_decision=activation_decision,
                    eos_token_id=eos_token_id,
                    step=step,
                    branch="attention_triggered_fsa_forced",
                )
            )

            if self._can_delay_constraint(forced_active):
                delay_base = base_hyp.clone()

                delay_constraint = self._find_matching_constraint(
                    delay_base.constraints,
                    forced_active,
                )

                delay_count = self._increment_delay_count(delay_constraint)

                result.extend(
                    self._expand_with_normal_logits(
                        base_hyp=delay_base,
                        logits=logits,
                        attention_info=attention_info,
                        activation_decision=activation_decision,
                        eos_token_id=eos_token_id,
                        step=step,
                        branch=f"delay_before_fsa_{delay_count}",
                        top_k=self.delay_branch_top_k,
                        extra_log_penalty=self.delay_penalty,
                    )
                )

            return self._dedupe_hypotheses(result)

        return self._expand_with_normal_logits(
            base_hyp=base_hyp,
            logits=logits,
            attention_info=attention_info,
            activation_decision=None,
            eos_token_id=eos_token_id,
            step=step,
            branch="normal_free_decode_no_constraint_active",
            top_k=self.top_k,
        )

    # --------------------------------------------------
    # Selection
    # --------------------------------------------------

    def _select_best(self, hypotheses: List[Hypothesis]) -> Optional[Hypothesis]:
        if not hypotheses:
            return None

        min_len = self._current_min_decode_length

        completed_long = [
            hyp
            for hyp in hypotheses
            if hyp.ended
            and self._all_forced_done(hyp)
            and self._generated_len(hyp) >= min_len
        ]

        if completed_long:
            return max(completed_long, key=self._selection_score)

        completed = [
            hyp
            for hyp in hypotheses
            if hyp.ended and self._all_forced_done(hyp)
        ]

        if completed:
            return max(completed, key=self._selection_score)

        valid_long = [
            hyp
            for hyp in hypotheses
            if self._all_forced_done(hyp)
            and self._generated_len(hyp) >= min_len
        ]

        if valid_long:
            return max(valid_long, key=self._selection_score)

        valid = [
            hyp
            for hyp in hypotheses
            if self._all_forced_done(hyp)
        ]

        if valid:
            return max(valid, key=self._selection_score)

        return max(
            hypotheses,
            key=lambda hyp: (
                self._forced_done_count(hyp),
                self._fsa_progress_sum(hyp),
                self._selection_score(hyp),
            ),
        )

    # --------------------------------------------------
    # Public decode
    # --------------------------------------------------

    def decode(
        self,
        encoder_outputs,
        attention_mask,
        constraints: Optional[List[Constraint]] = None,
    ) -> Dict[str, Any]:
        self.last_debug = {
            "steps": [],
            "stopped_reason": None,
        }

        attention_mask = self._prepare_attention_mask(attention_mask)

        source_len = self._source_len(attention_mask)
        self._current_source_len = source_len
        self._current_min_decode_length = self._compute_min_decode_length(source_len)

        self.last_debug.update(
            {
                "source_len": source_len,
                "min_decode_length": self._current_min_decode_length,
                "min_length_ratio": self.min_length_ratio,
                "short_output_penalty": self.short_output_penalty,
                "max_delay_before_constraint": self.max_delay_before_constraint,
                "delay_branch_top_k": self.delay_branch_top_k,
                "delay_penalty": self.delay_penalty,
                "covered_span_mask_enabled": getattr(
                    self.covered_span_masker,
                    "enable_source_mask",
                    None,
                ),
            }
        )

        init_hyp = Hypothesis(
            token_ids=[self._decoder_start_token_id()],
            score=0.0,
            constraints=self._clone_constraints(constraints or []),
            trace=[],
            ended=False,
        )

        beam = DBABeam(
            beam_size=self.beam_size,
            length_penalty=self.length_penalty,
            constraint_bonus=0.0,
            prefer_higher_banks=False,
            deduplicate=True,
        )

        beam.add(init_hyp)
        beam.prune()

        for step in range(1, self.max_length + 1):
            new_beam = DBABeam(
                beam_size=self.beam_size,
                length_penalty=self.length_penalty,
                constraint_bonus=0.0,
                prefer_higher_banks=False,
                deduplicate=True,
            )

            expanded_count = 0

            for hyp in beam:
                new_hyps = self._expand_one_hypothesis(
                    hyp=hyp,
                    encoder_outputs=encoder_outputs,
                    base_attention_mask=attention_mask,
                    step=step,
                )

                expanded_count += len(new_hyps)
                new_beam.extend(new_hyps)

            if new_beam.is_empty():
                self.last_debug["stopped_reason"] = "new_beam_empty"
                break

            new_beam.prune()
            beam = new_beam

            self.last_debug["steps"].append(
                {
                    "step": step,
                    "expanded_count": expanded_count,
                    "beam_size": len(beam),
                    "banks": beam.banks_summary(),
                    "beam_debug": beam.debug_info(),
                    "covered_span_masker": self.covered_span_masker.debug_info()
                    if hasattr(self.covered_span_masker, "debug_info")
                    else {},
                }
            )

            if beam.all_ended():
                self.last_debug["stopped_reason"] = "all_hypotheses_ended"
                break

        if self.last_debug.get("stopped_reason") is None:
            self.last_debug["stopped_reason"] = "max_length_reached"

        best = self._select_best(list(beam))

        if best is None:
            return {
                "translation": "",
                "generated_ids": [],
                "trace": [],
                "constraints": constraints or [],
                "beam_summary": {},
                "score": 0.0,
                "normalized_score": 0.0,
                "selection_score": 0.0,
                "short_output_penalty": 0.0,
                "generated_length": 0,
                "min_decode_length": self._current_min_decode_length,
                "source_len": int(self._current_source_len or 0),
                "done_count": 0,
                "forced_total": 0,
                "all_forced_done": True,
                "fsa_progress_sum": 0,
                "debug": dict(self.last_debug),
            }

        generated_ids = best.token_ids[1:]
        translation = self._decode_ids(generated_ids, skip_special_tokens=True)

        return {
            "translation": translation,
            "generated_ids": generated_ids,
            "trace": list(best.trace),
            "constraints": best.constraints,
            "beam_summary": beam.banks_summary(),
            "score": float(best.score),
            "normalized_score": float(best.normalized_score(self.length_penalty)),
            "selection_score": float(self._selection_score(best)),
            "short_output_penalty": float(self._short_penalty(best)),
            "generated_length": int(self._generated_len(best)),
            "min_decode_length": int(self._current_min_decode_length),
            "source_len": int(self._current_source_len or 0),
            "done_count": self._forced_done_count(best),
            "forced_total": self._forced_total_count(best),
            "all_forced_done": self._all_forced_done(best),
            "fsa_progress_sum": self._fsa_progress_sum(best),
            "debug": dict(self.last_debug),
        }

    def decode_from_input_ids(
        self,
        input_ids,
        attention_mask=None,
        constraints: Optional[List[Constraint]] = None,
    ) -> Dict[str, Any]:
        encoder_outputs, attention_mask = self.encode_source(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return self.decode(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            constraints=constraints,
        )


MultiStackBeamSearch = ConstrainedBeamSearch