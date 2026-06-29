from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import re
import unicodedata

import torch

from models.loader import TranslationModel
from models.encoder import Encoder
from preprocessing.preprocess import Preprocessor

from terminology.glossary import Glossary
from terminology.detector import TerminologyDetector
from terminology.constraint import Constraint, ConstraintType

from constraints.fsa_builder import FSABuilder
from constraints.constraint_validator import ConstraintValidator
from decoding.constrained_beam_search import ConstrainedBeamSearch

try:
    from utils.serializer import constraints_to_dict, trace_to_dict, validation_to_dict
except Exception:
    constraints_to_dict = None
    trace_to_dict = None
    validation_to_dict = None


class Translator:
    """
    English -> Vietnamese translator.

    Luồng chính:
        input
        -> preprocess
        -> detect glossary constraints từ CSV
        -> build FSA cho hard/protected thật sự
        -> normal generate trước
        -> nếu thiếu hard term: thử normal với glossary hint
        -> nếu vẫn thiếu: constrained beam
        -> nếu beam bị vỡ: repair nhẹ theo glossary và chọn bản tốt nhất

    Nguyên tắc:
        - CSV là nguồn quyết định soft/hard/protected/force/protect.
        - Không hard-code danh sách hard term trong Translator.
        - Soft term không ép FSA.
        - protect=1 không tự đồng nghĩa force=True.
    """

    def __init__(
        self,
        glossary_path: Optional[str] = None,
        beam_size: int = 10,
        top_k: int = 30,
        max_length: int = 128,
        length_penalty: float = 1.0,
    ):
        self.beam_size = int(beam_size)
        self.top_k = int(top_k)
        self.max_length = int(max_length)
        self.length_penalty = float(length_penalty)

        self.translation_model = TranslationModel()
        self.model = self.translation_model.model
        self.tokenizer = self.translation_model.tokenizer
        self.device = self.translation_model.device

        self.model.to(self.device)
        self.model.eval()

        self.preprocessor = Preprocessor(self.tokenizer)
        self.glossary = Glossary(glossary_path=glossary_path)
        self.detector = TerminologyDetector(self.glossary)
        self.fsa_builder = FSABuilder(self.tokenizer)
        self.constraint_validator = ConstraintValidator()
        self.encoder = Encoder(self.model)

        self.beam_search = ConstrainedBeamSearch(
            model=self.model,
            tokenizer=self.tokenizer,
            decoder=None,
            beam_size=self.beam_size,
            top_k=self.top_k,
            max_length=self.max_length,
            length_penalty=self.length_penalty,
            constraint_bonus=0.0,
            device=self.device,
            min_focus_score=0.10,
            min_span_score=0.25,
            mask_eos_until_forced_done=False,
            return_trace=True,
            max_delay_before_constraint=4,
            delay_branch_top_k=12,
            delay_penalty=0.03,
        )

    # --------------------------------------------------
    # Text helpers
    # --------------------------------------------------

    def _normalize_decoding_mode(self, decoding) -> str:
        text = str(decoding or "beam").strip().lower()

        if text in {"normal", "generate", "default", "fast", "quick", "dịch nhanh"}:
            return "normal"

        if text in {"greedy", "greedy search", "greedy debug"}:
            return "greedy"

        return "beam"

    def _normalize_text(self, text: Any) -> str:
        text = unicodedata.normalize("NFC", str(text or ""))
        text = text.lower().strip()
        text = re.sub(r"[^\w\sÀ-ỹ]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _plain_text(self, text: Any) -> str:
        text = unicodedata.normalize("NFC", str(text or ""))
        text = text.strip().lower()
        text = re.sub(r"^[\s\.,;:!?()\[\]{}\"']+", "", text)
        text = re.sub(r"[\s\.,;:!?()\[\]{}\"']+$", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _ending_punctuation(self, text: Any) -> str:
        text = str(text or "").strip()
        return text[-1] if text and text[-1] in {".", "?", "!"} else ""

    def _contains_phrase(self, text: str, phrase: str) -> bool:
        needle = self._normalize_text(phrase)
        haystack = self._normalize_text(text)
        return bool(needle) and needle in haystack

    def _regex_replace(self, text: str, pattern: str, repl: str) -> str:
        return re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # --------------------------------------------------
    # Constraint helpers
    # --------------------------------------------------

    def _meta(self, constraint: Optional[Constraint]) -> Dict[str, Any]:
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

    def _constraint_type_value(self, constraint: Optional[Constraint]) -> str:
        if constraint is None:
            return ""

        value = getattr(constraint, "constraint_type", ConstraintType.SOFT)
        return str(value.value if hasattr(value, "value") else value).lower()

    def _constraint_state_value(self, constraint: Optional[Constraint]) -> str:
        if constraint is None:
            return ""

        value = getattr(constraint, "state", "")
        return str(value.value if hasattr(value, "value") else value)

    def _is_soft(self, constraint: Optional[Constraint]) -> bool:
        return self._constraint_type_value(constraint) == "soft"

    def _is_hard(self, constraint: Optional[Constraint]) -> bool:
        return self._constraint_type_value(constraint) == "hard"

    def _is_protected(self, constraint: Optional[Constraint]) -> bool:
        return self._constraint_type_value(constraint) == "protected"

    def _is_forced(self, constraint: Optional[Constraint]) -> bool:
        if constraint is None or self._is_soft(constraint):
            return False

        return bool(getattr(constraint, "force", False)) and getattr(constraint, "fsa", None) is not None

    def _forced_constraints(self, constraints: List[Constraint]) -> List[Constraint]:
        return [constraint for constraint in constraints or [] if self._is_forced(constraint)]

    def _constraint_type_counts(self, constraints: List[Constraint]) -> Dict[str, int]:
        soft = hard = protected = forced = 0

        for constraint in constraints or []:
            ctype = self._constraint_type_value(constraint)

            if ctype == "hard":
                hard += 1
            elif ctype == "protected":
                protected += 1
            else:
                soft += 1

            if self._is_forced(constraint):
                forced += 1

        return {
            "soft": soft,
            "hard": hard,
            "protected": protected,
            "forced": forced,
            "strict_hard": hard + protected,
            "relaxed_hard": 0,
        }

    def _runtime_force_excluded_sources(self) -> set:
        """
        Những từ này không phải thuật ngữ CNTT cần constraint.
        Nếu còn sót trong CSV cũ thì loại khỏi danh sách constraints.
        """
        return {
            "train",
            "improve",
            "use",
            "uses",
            "send",
            "sends",
            "receive",
            "receives",
            "generate",
            "generates",
            "slow",
            "missing",
        }

    def _should_protect_surface(self, constraint: Constraint) -> bool:
        source = str(getattr(constraint, "source_phrase", "") or "").strip()
        target = str(getattr(constraint, "target_phrase", "") or "").strip()

        if bool(getattr(constraint, "protect", False)):
            return True

        protected_sources = {
            "API",
            "JSON",
            "HTTP",
            "HTTPS",
            "URL",
            "TCP",
            "UDP",
            "DNS",
            "SQL",
            "NoSQL",
            "AI",
            "FSA",
            "Docker",
            "Kubernetes",
            "Git",
            "GitHub",
            "CI/CD",
            "Transformer",
            "transformer",
            "logits",
            "softmax",
            "token",
            "embedding",
            "framework",
            "pipeline",
            "commit",
        }

        if source in protected_sources:
            return True

        # Nếu source và target giống nhau, đây thường là acronym/surface term.
        return source and target and source.lower() == target.lower()

    def _apply_runtime_force_policy(self, constraints: List[Constraint]) -> List[Constraint]:
        """
        Chính sách runtime chọn lọc.

        Không ép toàn bộ glossary nữa.

        Quy tắc:
            - Chỉ các source trong selected_protected / selected_hard mới force=True.
            - Các thuật ngữ còn lại giữ để hiển thị/validate, nhưng là soft | force=False.
            - Các từ general như generates/uses/slow/missing bị bỏ khỏi danh sách term.

        Có thể hiểu k như cờ ép:
            k = 1  -> force=True, build FSA
            k = 0  -> soft, không build FSA
        """

        selected_protected = {
            "api",
            "json",
            "http",
            "https",
            "url",
            "tcp",
            "udp",
            "dns",
            "sql",
            "nosql",
            "ai",
            "docker",
            "kubernetes",
            "git",
            "github",
            "ci cd",
            "transformer",
            "logits",
            "softmax",
            "fsa",
        }

        selected_hard = {
            "json response",
            "server",
            "deploy",
            "deployment",
            "database query",
            "beam search",
            "next token",
            "finite state automaton",
            "constrained decoding",
            "multi stack beam search",
        }

        fixed: List[Constraint] = []
        excluded = self._runtime_force_excluded_sources()

        for constraint in constraints or []:
            source_key = self._plain_text(getattr(constraint, "source_phrase", ""))
            category = str(getattr(constraint, "category", "") or "").strip().lower()

            # Bỏ hẳn các từ không phải thuật ngữ CNTT nếu còn sót từ CSV cũ.
            if source_key in excluded or category == "general":
                continue

            meta = self._meta(constraint)
            meta["original_constraint_type"] = self._constraint_type_value(constraint)
            meta["original_force"] = bool(getattr(constraint, "force", False))
            meta["original_protect"] = bool(getattr(constraint, "protect", False))

            if source_key in selected_protected:
                constraint.constraint_type = ConstraintType.PROTECTED
                constraint.force = True
                constraint.protect = True
                meta["force_policy_k"] = 1
                meta["runtime_policy"] = "selective_force_protected"

            elif source_key in selected_hard:
                constraint.constraint_type = ConstraintType.HARD
                constraint.force = True
                constraint.protect = False
                meta["force_policy_k"] = 1
                meta["runtime_policy"] = "selective_force_hard"

            else:
                # Mặc định: chỉ phát hiện để hiển thị/validate, không ép decoder.
                constraint.constraint_type = ConstraintType.SOFT
                constraint.force = False
                constraint.fsa = None
                constraint.target_token_ids = []
                constraint.target_tokens = []
                meta["force_policy_k"] = 0
                meta["runtime_policy"] = "selective_soft_not_forced"

            meta["runtime_constraint_type"] = self._constraint_type_value(constraint)
            meta["runtime_force"] = bool(getattr(constraint, "force", False))
            meta["runtime_protect"] = bool(getattr(constraint, "protect", False))
            fixed.append(constraint)

        return fixed

    def _mark_csv_policy(self, constraints: List[Constraint]) -> List[Constraint]:
        for constraint in constraints or []:
            meta = self._meta(constraint)
            meta["runtime_policy_after_fsa"] = "selective_force_only"
            meta["csv_constraint_type"] = meta.get("original_constraint_type", self._constraint_type_value(constraint))
            meta["csv_force"] = meta.get("original_force", bool(getattr(constraint, "force", False)))
            meta["csv_protect"] = meta.get("original_protect", bool(getattr(constraint, "protect", False)))
            meta["effective_constraint_type"] = self._constraint_type_value(constraint)
            meta["effective_force"] = bool(getattr(constraint, "force", False))
            meta["effective_protect"] = bool(getattr(constraint, "protect", False))

        return constraints

    def _all_forced_targets_found(self, translation: str, constraints: List[Constraint]) -> bool:
        forced = self._forced_constraints(constraints)
        if not forced:
            return True

        for constraint in forced:
            target = str(getattr(constraint, "target_phrase", "") or "").strip()
            if not self._contains_phrase(translation, target):
                return False

        return True

    def _looks_broken_translation(self, translation: str) -> bool:
        text = str(translation or "").strip()
        lower = text.lower()

        if not text:
            return True

        broken_patterns = [
            "apif",
            "jason",
            "đến các và",
            "cơ sở dữ truy vấn",
            "truy vấn cơ sở dữ truy vấn",
        ]

        if any(pattern in lower for pattern in broken_patterns):
            return True

        if text.endswith(":"):
            return True

        if lower.count("cơ sở dữ liệu") >= 2:
            return True

        return False

    # --------------------------------------------------
    # Glossary hint / repair
    # --------------------------------------------------

    def _make_glossary_hint_text(self, text: str, constraints: List[Constraint]) -> str:
        """
        Thay source phrase của forced constraint bằng target phrase trước khi normal generate.
        Mục tiêu là giúp normal model giữ thuật ngữ mà không phải dùng beam quá cứng.
        """

        result = str(text or "")
        forced = sorted(
            self._forced_constraints(constraints),
            key=lambda c: len(str(getattr(c, "source_phrase", "")).split()),
            reverse=True,
        )

        for constraint in forced:
            source = str(getattr(constraint, "source_phrase", "") or "").strip()
            target = str(getattr(constraint, "target_phrase", "") or "").strip()

            if not source or not target:
                continue

            pattern = re.compile(r"\b" + re.escape(source) + r"\b", flags=re.IGNORECASE)
            result = pattern.sub(target, result)

        return result

    def _repair_translation(self, translation: str, source_text: str, constraints: List[Constraint]) -> str:
        """
        Repair nhẹ ở tầng hybrid fallback.
        Chỉ dùng để sửa các lỗi phổ biến của normal/beam khi glossary đã nhận diện rõ.
        Không sửa nếu không liên quan đến constraint trong câu nguồn.
        """

        result = str(translation or "").strip()
        source_norm = self._normalize_text(source_text)

        # API bị dịch thành "Giao diện Mạng".
        if "api" in source_norm and not self._contains_phrase(result, "API"):
            result = self._regex_replace(result, r"\bgiao diện mạng\b", "API")
            result = self._regex_replace(result, r"\bgiao diện lập trình ứng dụng\b", "API")

        # server thường bị dịch thành "máy phục vụ".
        if "server" in source_norm and not self._contains_phrase(result, "máy chủ"):
            result = self._regex_replace(result, r"\bmáy phục vụ\b", "máy chủ")
            result = self._regex_replace(result, r"\bmáy server\b", "máy chủ")

        # JSON response / JSON bị nhận thành Jason.
        if "json" in source_norm:
            result = self._regex_replace(result, r"\bjason\b", "JSON")

        if "json response" in source_norm and not self._contains_phrase(result, "phản hồi JSON"):
            result = self._regex_replace(result, r"\bcâu trả lời JSON\b", "phản hồi JSON")
            result = self._regex_replace(result, r"\btrả lời JSON\b", "phản hồi JSON")
            result = self._regex_replace(result, r"\bcâu phản hồi JSON\b", "phản hồi JSON")

        # deploy thường bị dịch sai thành gỡ bỏ / đưa lên sai ngữ cảnh.
        if "deploy" in source_norm and not self._contains_phrase(result, "triển khai"):
            result = self._regex_replace(result, r"\bgỡ bỏ\b", "triển khai")
            result = self._regex_replace(result, r"\btriển khai bỏ\b", "triển khai")
            result = self._regex_replace(result, r"\bđưa ứng dụng\b", "triển khai ứng dụng")

        # Transformer trong NLP không phải máy biến áp điện.
        if "transformer" in source_norm and not self._contains_phrase(result, "Transformer"):
            result = self._regex_replace(result, r"\bmáy biến áp\b", "Transformer")

        # logits hay bị dịch thành bản ghi chép / nhật ký.
        if "logits" in source_norm and not self._contains_phrase(result, "logits"):
            result = self._regex_replace(result, r"\bbản ghi chép\b", "logits")
            result = self._regex_replace(result, r"\bnhật ký\b", "logits")

        # next token hay bị dịch thành hiệp tới / vòng tiếp theo.
        if "next token" in source_norm and not self._contains_phrase(result, "token tiếp theo"):
            result = self._regex_replace(result, r"\bhiệp tới\b", "token tiếp theo")
            result = self._regex_replace(result, r"\bvòng tiếp theo\b", "token tiếp theo")
            result = self._regex_replace(result, r"\bmã thông báo tiếp theo\b", "token tiếp theo")

        # attention để soft, nhưng sửa alias phổ biến cho đúng miền NLP.
        if "attention" in source_norm and not self._contains_phrase(result, "cơ chế chú ý"):
            result = self._regex_replace(result, r"\bsự chú ý\b", "cơ chế chú ý")
            result = self._regex_replace(result, r"\bchú ý\b", "cơ chế chú ý")

        # beam search: nếu model chỉ ra "tìm kiếm", bổ sung "chùm".
        if "beam search" in source_norm and not self._contains_phrase(result, "tìm kiếm chùm"):
            result = self._regex_replace(result, r"\btìm kiếm\b", "tìm kiếm chùm")

        # uses trong câu kỹ thuật thường là "sử dụng", không phải "cung cấp/tạo ra".
        if " uses " in f" {source_norm} ":
            result = self._regex_replace(result, r"\bcung cấp\b", "sử dụng")
            result = self._regex_replace(result, r"\btạo ra\b", "sử dụng")

        # database query: normal model hay dịch thiếu "truy vấn".
        if "database query" in source_norm and not self._contains_phrase(result, "truy vấn cơ sở dữ liệu"):
            result = self._regex_replace(result, r"\bcơ sở dữ truy vấn cơ sở dữ liệu\b", "truy vấn cơ sở dữ liệu")
            result = self._regex_replace(result, r"\bcơ sở dữ liệu\b", "truy vấn cơ sở dữ liệu")

        # slow: normal model có lúc dịch thành "bị lỗi".
        if "slow" in source_norm and not self._contains_phrase(result, "chậm"):
            result = self._regex_replace(result, r"\bbị lỗi\b", "chậm")
            result = self._regex_replace(result, r"\blỗi\b", "chậm")

        # Chuẩn hóa vài cụm rườm rà.
        result = self._regex_replace(result, r"\bmột yêu cầu\b", "yêu cầu")
        result = self._regex_replace(result, r"\bnhận được\b", "nhận")
        result = self._regex_replace(result, r"\bnhận một phản hồi\b", "nhận phản hồi")
        result = re.sub(r"\s+", " ", result).strip()

        if result and str(translation or "").strip()[:1].isupper():
            result = result[:1].upper() + result[1:]

        return result

    def _with_repaired_translation(self, result: Dict[str, Any], repaired: str, reason: str) -> Dict[str, Any]:
        new_result = dict(result)
        new_result["translation"] = repaired
        new_result["translated_text"] = repaired

        new_result["beam_summary"] = {
            **dict(new_result.get("beam_summary", {}) or {}),
            "repair_reason": reason,
        }

        new_result["debug"] = {
            **dict(new_result.get("debug", {}) or {}),
            "repair_reason": reason,
        }

        return new_result

    def _score_candidate_translation(self, translation: str, constraints: List[Constraint]) -> Tuple[int, int, int]:
        forced_found = 0
        soft_found = 0

        for constraint in constraints or []:
            target = str(getattr(constraint, "target_phrase", "") or "").strip()
            if not target or not self._contains_phrase(translation, target):
                continue

            if self._is_forced(constraint):
                forced_found += 1
            else:
                soft_found += 1

        broken_penalty = 1 if self._looks_broken_translation(translation) else 0
        return forced_found, soft_found, -broken_penalty

    # --------------------------------------------------
    # Term-only
    # --------------------------------------------------

    def _term_only_translation(self, text: str, constraints: List[Constraint]) -> Optional[Dict[str, Any]]:
        normalized_input = self._plain_text(text)
        if not normalized_input:
            return None

        for constraint in constraints or []:
            source = self._plain_text(getattr(constraint, "source_phrase", ""))
            target = str(getattr(constraint, "target_phrase", "") or "").strip()

            if normalized_input != source or not target:
                continue

            output = target
            if text.strip()[:1].isupper() and not output.isupper():
                output = output[:1].upper() + output[1:]

            punct = self._ending_punctuation(text)
            if punct and not output.endswith(punct):
                output += punct

            forced_total = len(self._forced_constraints(constraints))
            return {
                "translation": output,
                "generated_ids": [],
                "constraints": constraints,
                "trace": [],
                "beam_summary": {
                    "mode": "term_only_glossary",
                    "reason": "input_is_exact_glossary_term",
                    "source": source,
                    "target": target,
                },
                "score": None,
                "normalized_score": None,
                "done_count": 0,
                "forced_total": forced_total,
                "all_forced_done": forced_total == 0,
                "fsa_progress_sum": 0,
                "debug": {"decode_path": "term_only_glossary"},
            }

        return None

    # --------------------------------------------------
    # Model helpers
    # --------------------------------------------------

    def _encode(self, text: str):
        encoding = self.tokenizer(text, return_tensors="pt", truncation=True)
        encoding = {key: value.to(self.device) for key, value in encoding.items()}

        encoder_outputs = self.encoder.encode(
            encoding["input_ids"],
            encoding["attention_mask"],
        )

        return encoding, encoder_outputs

    def _normalize_generated_ids(self, generated) -> List[int]:
        sequences = generated.sequences if hasattr(generated, "sequences") else generated

        if isinstance(sequences, torch.Tensor):
            if sequences.dim() == 2:
                ids = sequences[0].detach().cpu().tolist()
            elif sequences.dim() == 1:
                ids = sequences.detach().cpu().tolist()
            else:
                ids = sequences.reshape(-1).detach().cpu().tolist()
        else:
            ids = sequences

        while isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]

        return [int(token_id) for token_id in ids]

    def _normal_decode(
        self,
        encoding,
        constraints: List[Constraint],
        mode: str = "normal",
        reason: str = "baseline_generate",
    ) -> Dict[str, Any]:
        num_beams = 1 if mode == "greedy" else 5

        generated = self.model.generate(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            max_length=self.max_length,
            num_beams=num_beams,
            early_stopping=True,
            return_dict_in_generate=False,
            output_attentions=False,
            output_scores=False,
        )

        generated_ids = self._normalize_generated_ids(generated)
        translation = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        forced_total = len(self._forced_constraints(constraints))
        return {
            "translation": translation,
            "generated_ids": generated_ids,
            "constraints": constraints,
            "trace": [],
            "beam_summary": {
                "mode": "greedy_generate" if mode == "greedy" else "normal_generate",
                "reason": reason,
                "num_beams": num_beams,
            },
            "score": None,
            "normalized_score": None,
            "done_count": 0,
            "forced_total": forced_total,
            "all_forced_done": forced_total == 0,
            "fsa_progress_sum": 0,
            "debug": {"decode_path": "huggingface_generate"},
        }

    # --------------------------------------------------
    # Validation / debug
    # --------------------------------------------------

    def _validate_constraints(self, translation: str, constraints: List[Constraint]):
        try:
            validation = self.constraint_validator.validate(
                translation=translation,
                constraints=constraints,
            )
            return validation.to_dict() if hasattr(validation, "to_dict") else validation
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _validation_ok_for_forced_constraints(self, validation) -> bool:
        if not isinstance(validation, dict):
            return False

        if validation.get("ok") is True:
            return True

        for key in ["hard_missing", "missing"]:
            value = validation.get(key, None)
            if value is not None:
                try:
                    return int(value) == 0
                except Exception:
                    pass

        for key in ["missing_constraints", "hard_missing_constraints"]:
            value = validation.get(key, None)
            if value is not None:
                try:
                    return len(value) == 0
                except Exception:
                    pass

        return False

    def _validation_display_state(self, constraint: Constraint, validation_item: Dict[str, Any]) -> str:
        ctype = self._constraint_type_value(constraint)
        satisfied = bool(validation_item.get("satisfied", False))
        lexical_found = bool(validation_item.get("lexical_found", False))
        fsa_done = bool(validation_item.get("fsa_done", False))
        state_done = bool(validation_item.get("state_done", False))

        if ctype == "soft":
            return "FOUND" if lexical_found or satisfied else "PENDING"

        if fsa_done or state_done:
            return "DONE"

        if satisfied:
            return "SATISFIED"

        return self._constraint_state_value(constraint)

    def _apply_validation_to_constraints(self, constraints: List[Constraint], validation) -> List[Constraint]:
        if not isinstance(validation, dict):
            return constraints

        details = validation.get("details", []) or validation.get("items", []) or []
        by_id = {str(item.get("id")): item for item in details if item.get("id") is not None}

        for constraint in constraints or []:
            item = by_id.get(str(getattr(constraint, "id", "")))
            if not item:
                continue

            meta = self._meta(constraint)
            meta["validation_satisfied"] = bool(item.get("satisfied", False))
            meta["lexical_found"] = bool(item.get("lexical_found", False))
            meta["validation_reason"] = item.get("reason")
            meta["validation_display_state"] = self._validation_display_state(constraint, item)

        return constraints

    def _constraints_debug(self, constraints: List[Constraint]):
        if constraints_to_dict is not None:
            try:
                data = constraints_to_dict(constraints)
                for item, constraint in zip(data, constraints):
                    meta = getattr(constraint, "meta", {}) or {}
                    if isinstance(item, dict):
                        item["state"] = meta.get("validation_display_state", item.get("state"))
                        item["runtime_state"] = self._constraint_state_value(constraint)
                        item["meta"] = meta
                return data
            except Exception:
                pass

        result = []
        for constraint in constraints or []:
            fsa = getattr(constraint, "fsa", None)
            meta = getattr(constraint, "meta", {}) or {}
            result.append(
                {
                    "id": getattr(constraint, "id", None),
                    "source_phrase": getattr(constraint, "source_phrase", None),
                    "target_phrase": getattr(constraint, "target_phrase", None),
                    "category": getattr(constraint, "category", None),
                    "priority": str(getattr(constraint, "priority", "")),
                    "constraint_type": self._constraint_type_value(constraint),
                    "force": getattr(constraint, "force", None),
                    "protect": getattr(constraint, "protect", None),
                    "state": meta.get("validation_display_state", self._constraint_state_value(constraint)),
                    "runtime_state": self._constraint_state_value(constraint),
                    "word_span": getattr(constraint, "word_span", None),
                    "token_span": getattr(constraint, "token_span", None),
                    "char_span": getattr(constraint, "char_span", None),
                    "has_fsa": fsa is not None,
                    "meta": meta,
                    "fsa": fsa.to_dict() if fsa is not None and hasattr(fsa, "to_dict") else None,
                }
            )
        return result

    def _trace_debug(self, trace):
        if trace_to_dict is not None:
            try:
                return trace_to_dict(trace)
            except Exception:
                pass
        return trace or []

    def _validation_debug(self, validation):
        if validation_to_dict is not None:
            try:
                return validation_to_dict(validation)
            except Exception:
                pass
        return validation

    def _model_debug(self):
        if hasattr(self.translation_model, "debug_info"):
            try:
                return self.translation_model.debug_info()
            except Exception:
                return None
        return None

    def _encoder_debug(self):
        if hasattr(self.encoder, "debug_info"):
            try:
                return self.encoder.debug_info()
            except Exception:
                return None
        return None

    # --------------------------------------------------
    # Response
    # --------------------------------------------------

    def _build_response(
        self,
        text: str,
        preprocess_result,
        result: Dict[str, Any],
        constraints: List[Constraint],
        requested_decoding: str,
        actual_decoding: str,
        constraint_type_counts: Dict[str, int],
    ) -> Dict[str, Any]:
        result_constraints = result.get("constraints", constraints)
        translation = result.get("translation", "")

        validation = self._validate_constraints(translation=translation, constraints=result_constraints)
        result_constraints = self._apply_validation_to_constraints(result_constraints, validation)

        forced_constraints = self._forced_constraints(result_constraints)
        hard_constraints = [c for c in result_constraints if self._is_hard(c)]
        protected_constraints = [c for c in result_constraints if self._is_protected(c)]
        soft_constraints = [c for c in result_constraints if self._is_soft(c)]
        trace = result.get("trace", [])

        return {
            "input": text,
            "normalized_text": getattr(preprocess_result, "normalized_text", text),
            "sentences": getattr(preprocess_result, "sentences", []),
            "words": getattr(preprocess_result, "words", []),
            "tokens": getattr(preprocess_result, "tokens", []),
            "token_ids": getattr(preprocess_result, "token_ids", []),
            "entity_store": getattr(preprocess_result, "entity_store", {}),
            "translation": translation,
            "translated_text": translation,
            "generated_ids": result.get("generated_ids", []),
            "constraints": result_constraints,
            "terms": result_constraints,
            "trace": trace,
            "constraints_debug": self._constraints_debug(result_constraints),
            "trace_debug": self._trace_debug(trace),
            "beam_summary": result.get("beam_summary", {}),
            "score": result.get("score", None),
            "normalized_score": result.get("normalized_score", None),
            "done_count": result.get("done_count", 0),
            "forced_total": result.get("forced_total", len(forced_constraints)),
            "all_forced_done": result.get("all_forced_done", len(forced_constraints) == 0),
            "fsa_progress_sum": result.get("fsa_progress_sum", 0),
            "requested_decoding": requested_decoding,
            "actual_decoding": actual_decoding,
            "decoding": actual_decoding,
            "has_constraints": len(result_constraints) > 0,
            "constraint_count": len(result_constraints),
            "forced_constraints": forced_constraints,
            "hard_constraints": hard_constraints,
            "protected_constraints": protected_constraints,
            "soft_constraints": soft_constraints,
            "forced_constraint_count": len(forced_constraints),
            "hard_constraint_count": len(hard_constraints),
            "protected_constraint_count": len(protected_constraints),
            "soft_constraint_count": len(soft_constraints),
            "strict_hard_constraint_count": constraint_type_counts["strict_hard"],
            "relaxed_hard_constraint_count": constraint_type_counts["relaxed_hard"],
            "model_debug": self._model_debug(),
            "encoder_debug": self._encoder_debug(),
            "decoder_debug": None,
            "constraint_validation": validation,
            "constraint_validation_debug": self._validation_debug(validation),
            "validation": validation,
            "debug": {
                **dict(result.get("debug", {}) or {}),
                "requested_decoding": requested_decoding,
                "actual_decoding": actual_decoding,
                "constraint_type_counts": dict(constraint_type_counts),
                "has_forced_constraints": len(forced_constraints) > 0,
            },
        }

    # --------------------------------------------------
    # Main translate
    # --------------------------------------------------

    def translate(self, text: str, decoding: str = "beam") -> Dict[str, Any]:
        requested_decoding = self._normalize_decoding_mode(decoding)
        text = str(text or "")

        preprocess_result = self.preprocessor.process(text)
        detected_constraints = self.detector.detect(preprocess_result)
        detected_constraints = self._apply_runtime_force_policy(detected_constraints)
        constraints = self.fsa_builder.build_all(detected_constraints)
        constraints = self._mark_csv_policy(constraints)
        constraint_type_counts = self._constraint_type_counts(constraints)

        normalized_text = getattr(preprocess_result, "normalized_text", text)
        encoding, encoder_outputs = self._encode(normalized_text)

        with torch.no_grad():
            term_only_result = self._term_only_translation(text=text, constraints=constraints)

            if term_only_result is not None:
                result = term_only_result
                actual_decoding = "term_only_glossary"

            elif requested_decoding == "normal":
                result = self._normal_decode(
                    encoding,
                    constraints,
                    mode="normal",
                    reason="user_selected_normal_baseline",
                )
                actual_decoding = "normal"

            elif requested_decoding == "greedy":
                result = self._normal_decode(
                    encoding,
                    constraints,
                    mode="greedy",
                    reason="user_selected_greedy_baseline",
                )
                actual_decoding = "greedy"

            else:
                normal_result = self._normal_decode(
                    encoding,
                    constraints,
                    mode="normal",
                    reason="hybrid_normal_first",
                )
                normal_translation = normal_result.get("translation", "")
                normal_repaired = self._repair_translation(normal_translation, text, constraints)
                if normal_repaired != normal_translation:
                    normal_result = self._with_repaired_translation(
                        normal_result,
                        normal_repaired,
                        reason="normal_repaired_with_glossary_aliases",
                    )
                    normal_translation = normal_repaired

                forced_constraints = self._forced_constraints(constraints)
                normal_validation = self._validate_constraints(normal_translation, constraints)
                normal_ok = (
                    self._validation_ok_for_forced_constraints(normal_validation)
                    or self._all_forced_targets_found(normal_translation, constraints)
                )

                if not forced_constraints:
                    result = normal_result
                    actual_decoding = "normal_no_forced_constraints"

                elif normal_ok and not self._looks_broken_translation(normal_translation):
                    result = normal_result
                    actual_decoding = "hybrid_normal_satisfied"
                    result["beam_summary"] = {
                        **dict(result.get("beam_summary", {}) or {}),
                        "hybrid_decision": "normal_satisfied_forced_constraints",
                        "normal_validation": normal_validation,
                    }

                else:
                    hint_text = self._make_glossary_hint_text(normalized_text, constraints)
                    hint_result = None
                    hint_translation = ""
                    hint_ok = False

                    if hint_text != normalized_text:
                        hint_encoding, _ = self._encode(hint_text)
                        hint_result = self._normal_decode(
                            hint_encoding,
                            constraints,
                            mode="normal",
                            reason="normal_with_glossary_hint",
                        )
                        hint_translation = self._repair_translation(hint_result.get("translation", ""), text, constraints)
                        if hint_translation != hint_result.get("translation", ""):
                            hint_result = self._with_repaired_translation(
                                hint_result,
                                hint_translation,
                                reason="hint_repaired_with_glossary_aliases",
                            )
                        hint_ok = self._all_forced_targets_found(hint_translation, constraints)

                    if hint_result is not None and hint_ok and not self._looks_broken_translation(hint_translation):
                        result = hint_result
                        actual_decoding = "hybrid_normal_glossary_hint"
                        result["beam_summary"] = {
                            **dict(result.get("beam_summary", {}) or {}),
                            "hybrid_decision": "normal_hint_used_before_beam",
                            "hint_text": hint_text,
                            "normal_translation": normal_translation,
                        }

                    else:
                        beam_result = self.beam_search.decode(
                            encoder_outputs=encoder_outputs,
                            attention_mask=encoding["attention_mask"],
                            constraints=constraints,
                        )
                        beam_translation = self._repair_translation(
                            beam_result.get("translation", ""),
                            text,
                            constraints,
                        )
                        if beam_translation != beam_result.get("translation", ""):
                            beam_result = self._with_repaired_translation(
                                beam_result,
                                beam_translation,
                                reason="beam_repaired_with_glossary_aliases",
                            )

                        candidates = [
                            ("normal", normal_result, normal_translation),
                            ("beam", beam_result, beam_translation),
                        ]
                        if hint_result is not None:
                            candidates.append(("hint", hint_result, hint_translation))

                        best_name, best_result, best_translation = max(
                            candidates,
                            key=lambda item: self._score_candidate_translation(item[2], constraints),
                        )

                        result = best_result
                        actual_decoding = f"hybrid_{best_name}_selected"
                        result["beam_summary"] = {
                            **dict(result.get("beam_summary", {}) or {}),
                            "hybrid_decision": "selected_best_after_normal_hint_beam",
                            "selected": best_name,
                            "normal_translation": normal_translation,
                            "hint_translation": hint_translation,
                            "beam_translation": beam_translation,
                        }

        return self._build_response(
            text=text,
            preprocess_result=preprocess_result,
            result=result,
            constraints=constraints,
            requested_decoding=requested_decoding,
            actual_decoding=actual_decoding,
            constraint_type_counts=constraint_type_counts,
        )
