from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
from typing import Any, Dict, Iterable, Iterator, List, Optional

from terminology.constraint import (
    ConstraintPriority,
    ConstraintType,
    normalize_constraint_type,
    normalize_source_key,
    normalize_space,
    parse_bool,
    priority_score,
)


@dataclass
class GlossaryTerm:
    """
    Một dòng thuật ngữ trong file CSV.

    Hỗ trợ CSV cũ:
        source,target,category,priority,protect

    Hỗ trợ CSV mới:
        source,target,category,priority,constraint_type,force,protect,alternatives,note

    constraint_type:
        soft
        hard
        protected

    Mặc định:
        soft, force=False

    Lý do:
        Không phải thuật ngữ nào cũng nên ép cứng bằng FSA.
        Nếu ép mọi từ thành HARD, câu dịch dễ bị đảo hoặc cụt.
    """

    source: str
    target: str
    category: str = "general"
    priority: ConstraintPriority = ConstraintPriority.MEDIUM
    constraint_type: ConstraintType = ConstraintType.SOFT
    force: bool = False
    protect: bool = False
    alternatives: List[str] = field(
        default_factory=list
    )
    meta: Dict[str, Any] = field(
        default_factory=dict
    )

    def source_key(
        self,
    ) -> str:
        return normalize_source_key(
            self.source
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "category": self.category,
            "priority": self.priority.value
            if hasattr(
                self.priority,
                "value",
            )
            else str(
                self.priority
            ),
            "constraint_type": self.constraint_type.value
            if hasattr(
                self.constraint_type,
                "value",
            )
            else str(
                self.constraint_type
            ),
            "force": bool(
                self.force
            ),
            "protect": bool(
                self.protect
            ),
            "alternatives": list(
                self.alternatives
            ),
            "meta": dict(
                self.meta
            ),
        }


class Glossary:
    """
    Đọc glossary từ:

        resources/terminology.csv
        hoặc
        resources/it.csv

    Bản này tương thích CSV cũ và CSV mới.

    Nếu CSV không có cột constraint_type:
        protect=1  -> protected
        protect=0  -> soft

    Nếu muốn ép thuật ngữ bằng FSA:
        ghi rõ constraint_type=hard,force=1

    Nếu chỉ muốn hiển thị/validate:
        constraint_type=soft
    """

    DEFAULT_FILES = [
        "resources/terminology.csv",
        "resources/it.csv",
    ]

    def __init__(
        self,
        glossary_path: Optional[str] = None,
    ):
        if glossary_path is None:
            glossary_path = self._default_glossary_path()

        self.path = Path(
            glossary_path
        )

        self.terms: Dict[str, GlossaryTerm] = {}

        self.load(
            self.path
        )

    # --------------------------------------------------
    # Path
    # --------------------------------------------------

    def _default_glossary_path(
        self,
    ) -> Path:
        root_dir = Path(
            __file__
        ).resolve().parent.parent

        candidates = [
            root_dir / "resources" / "terminology.csv",
            root_dir / "resources" / "it.csv",
        ]

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(
            "Không tìm thấy file thuật ngữ. "
            "Hãy tạo resources/terminology.csv hoặc resources/it.csv"
        )

    # --------------------------------------------------
    # Normalize / parse
    # --------------------------------------------------

    def _normalize_key(
        self,
        source: str,
    ) -> str:
        return normalize_source_key(
            source
        )

    def _get(
        self,
        row: Dict[str, Any],
        *names: str,
        default: str = "",
    ) -> str:
        for name in names:
            if name in row and row.get(
                name
            ) is not None:
                return str(
                    row.get(
                        name
                    )
                ).strip()

        return default

    def _parse_priority(
        self,
        value: Any,
    ) -> ConstraintPriority:
        raw = str(
            value or "medium"
        ).strip().lower()

        mapping = {
            "critical": ConstraintPriority.CRITICAL,
            "high": ConstraintPriority.HIGH,
            "medium": ConstraintPriority.MEDIUM,
            "low": ConstraintPriority.LOW,
        }

        return mapping.get(
            raw,
            ConstraintPriority.MEDIUM,
        )

    def _parse_alternatives(
        self,
        value: Any,
    ) -> List[str]:
        if value is None:
            return []

        text = str(
            value
        ).strip()

        if not text:
            return []

        result = []

        for chunk in text.replace(
            ";",
            "|",
        ).split(
            "|"
        ):
            chunk = normalize_space(
                chunk
            )

            if chunk:
                result.append(
                    chunk
                )

        return result

    def _resolve_constraint_type(
        self,
        row: Dict[str, Any],
        force: bool,
        protect: bool,
    ) -> ConstraintType:
        raw_type = self._get(
            row,
            "constraint_type",
            "type",
            "mode",
            default="",
        )

        if raw_type:
            return normalize_constraint_type(
                raw_type,
                force=force,
                protect=protect,
            )

        if protect:
            return ConstraintType.PROTECTED

        return ConstraintType.SOFT

    def _resolve_force(
        self,
        row: Dict[str, Any],
        constraint_type: ConstraintType,
    ) -> bool:
        """
        Chỉ force khi CSV có force=1/true/yes.

        Không tự suy luận:
            constraint_type=hard      -> force=True
            constraint_type=protected -> force=True

        Lý do: ta cần chính sách ép có chọn lọc.
        """

        raw_force = self._get(
            row,
            "force",
            "required",
            default="0",
        )

        return parse_bool(
            raw_force,
            default=False,
        )

    # --------------------------------------------------
    # Build term
    # --------------------------------------------------

    def _term_from_row(
        self,
        row: Dict[str, Any],
    ) -> Optional[GlossaryTerm]:
        source = normalize_space(
            self._get(
                row,
                "source",
                "src",
                "english",
                default="",
            )
        )

        target = normalize_space(
            self._get(
                row,
                "target",
                "tgt",
                "vietnamese",
                default="",
            )
        )

        if not source or not target:
            return None

        category = self._get(
            row,
            "category",
            "domain",
            default="general",
        ) or "general"

        priority = self._parse_priority(
            self._get(
                row,
                "priority",
                default="medium",
            )
        )

        protect = parse_bool(
            self._get(
                row,
                "protect",
                "protected",
                default="0",
            ),
            default=False,
        )

        raw_force = self._get(
            row,
            "force",
            "required",
            "hard",
            default="",
        )

        force_hint = (
            parse_bool(
                raw_force,
                default=False,
            )
            if raw_force
            else False
        )

        constraint_type = self._resolve_constraint_type(
            row=row,
            force=force_hint,
            protect=protect,
        )

        force = self._resolve_force(
            row=row,
            constraint_type=constraint_type,
        )

        if constraint_type == ConstraintType.PROTECTED:
            protect = True

        if not force:
            # Type có thể là hard/protected để hiển thị chính sách CSV,
            # nhưng FSA chỉ được build khi force=True.
            force = False

        alternatives = self._parse_alternatives(
            self._get(
                row,
                "alternatives",
                "alt",
                default="",
            )
        )

        meta = {
            "raw_row": dict(
                row
            ),
            "note": self._get(
                row,
                "note",
                "description",
                default="",
            ),
        }

        return GlossaryTerm(
            source=source,
            target=target,
            category=category,
            priority=priority,
            constraint_type=constraint_type,
            force=force,
            protect=protect,
            alternatives=alternatives,
            meta=meta,
        )

    # --------------------------------------------------
    # Duplicate resolution
    # --------------------------------------------------

    def _constraint_type_score(
        self,
        constraint_type: ConstraintType,
    ) -> int:
        if constraint_type == ConstraintType.PROTECTED:
            return 3

        if constraint_type == ConstraintType.HARD:
            return 2

        return 1

    def _should_replace_existing(
        self,
        old_term: GlossaryTerm,
        new_term: GlossaryTerm,
    ) -> bool:
        old_score = priority_score(
            old_term.priority
        )

        new_score = priority_score(
            new_term.priority
        )

        if new_score > old_score:
            return True

        if new_score < old_score:
            return False

        old_type_score = self._constraint_type_score(
            old_term.constraint_type
        )

        new_type_score = self._constraint_type_score(
            new_term.constraint_type
        )

        if new_type_score > old_type_score:
            return True

        if new_type_score < old_type_score:
            return False

        if bool(
            new_term.force
        ) and not bool(
            old_term.force
        ):
            return True

        if bool(
            new_term.protect
        ) and not bool(
            old_term.protect
        ):
            return True

        return len(
            new_term.source.split()
        ) > len(
            old_term.source.split()
        )

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    def load(
        self,
        path,
    ) -> None:
        path = Path(
            path
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file glossary: {path}"
            )

        self.terms.clear()

        with open(
            path,
            encoding="utf-8",
            newline="",
        ) as file:
            reader = csv.DictReader(
                file
            )

            if reader.fieldnames is None:
                raise ValueError(
                    "File glossary rỗng hoặc không có header."
                )

            fieldnames = {
                name.strip()
                for name in reader.fieldnames
                if name
            }

            if "source" not in fieldnames or "target" not in fieldnames:
                raise ValueError(
                    "File glossary thiếu cột bắt buộc: source,target"
                )

            for row in reader:
                term = self._term_from_row(
                    row
                )

                if term is None:
                    continue

                key = self._normalize_key(
                    term.source
                )

                if key not in self.terms:
                    self.terms[
                        key
                    ] = term
                    continue

                old_term = self.terms[
                    key
                ]

                if self._should_replace_existing(
                    old_term,
                    term,
                ):
                    self.terms[
                        key
                    ] = term

    # --------------------------------------------------
    # Accessors
    # --------------------------------------------------

    def items(
        self,
    ) -> Iterable[GlossaryTerm]:
        return sorted(
            self.terms.values(),
            key=lambda term: (
                len(
                    term.source.split()
                ),
                len(
                    term.source
                ),
            ),
            reverse=True,
        )

    def values(
        self,
    ) -> Iterable[GlossaryTerm]:
        return self.items()

    def keys(
        self,
    ) -> Iterable[str]:
        return self.terms.keys()

    def get(
        self,
        source_phrase: str,
        default=None,
    ):
        term = self.get_term(
            source_phrase
        )

        if term is None:
            return default

        return term.target

    def get_term(
        self,
        source_phrase: str,
    ) -> Optional[GlossaryTerm]:
        key = self._normalize_key(
            source_phrase
        )

        return self.terms.get(
            key
        )

    def to_list(
        self,
    ) -> List[Dict[str, Any]]:
        return [
            term.to_dict()
            for term in self.items()
        ]

    # --------------------------------------------------
    # Type filters
    # --------------------------------------------------

    def protected_terms(
        self,
    ) -> List[GlossaryTerm]:
        return [
            term
            for term in self.items()
            if term.constraint_type == ConstraintType.PROTECTED
            or term.protect
        ]

    def hard_terms(
        self,
    ) -> List[GlossaryTerm]:
        return [
            term
            for term in self.items()
            if term.constraint_type == ConstraintType.HARD
        ]

    def soft_terms(
        self,
    ) -> List[GlossaryTerm]:
        return [
            term
            for term in self.items()
            if term.constraint_type == ConstraintType.SOFT
        ]

    def forced_terms(
        self,
    ) -> List[GlossaryTerm]:
        return [
            term
            for term in self.items()
            if term.force
        ]

    # --------------------------------------------------
    # Python protocol
    # --------------------------------------------------

    def __len__(
        self,
    ) -> int:
        return len(
            self.terms
        )

    def __contains__(
        self,
        source_phrase: str,
    ) -> bool:
        return self._normalize_key(
            source_phrase
        ) in self.terms

    def __iter__(
        self,
    ) -> Iterator[GlossaryTerm]:
        return iter(
            self.items()
        )