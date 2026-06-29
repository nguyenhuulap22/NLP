from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Any
import copy


class FSAState(str, Enum):
    """
    Trạng thái nội bộ của FSA.

    IDLE:
        FSA đã được build nhưng chưa được attention kích hoạt.

    FORCING:
        FSA đang ép lần lượt target_token_ids.

    DONE:
        Đã sinh xong toàn bộ target_token_ids.

    FAILED:
        Đang ép nhưng token sinh ra không khớp.
        Trong beam search chuẩn, hypothesis này nên bị loại.
    """

    IDLE = "IDLE"
    FORCING = "FORCING"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass
class ConstraintFSA:
    """
    Finite-State Automaton cho một target phrase.

    Đây là FSA dùng trong lúc decode, không phải hậu xử lý.

    Cách dùng chuẩn trong decoding loop:

        1. ConstraintActivator đọc cross-attention.
        2. Nếu attention đang nhìn vào source_span của constraint:
              constraint.activate()
              constraint.fsa.activate()

        3. LogitsMasker đọc:
              allowed = fsa.allowed_token_ids()

           Nếu allowed không rỗng:
              mask toàn bộ vocab, chỉ giữ allowed token.

        4. Sau khi chọn token:
              ok = fsa.step(token_id)

        5. Nếu fsa.is_done:
              constraint.mark_done()
              covered_span_masker đánh dấu source span đã dịch.

    Điểm quan trọng:
        - FSA không tự quyết định khi nào active.
        - FSA không dùng source_order.
        - FSA không validate text sau dịch.
        - FSA chỉ biết target_token_ids và position hiện tại.

    Về extra token:
        Bài Hasler có nói có thể cho extra token trước/sau constraint
        khi decoder vẫn attention vào source span. Logic đó KHÔNG đặt
        cứng trong FSA. Nó được điều khiển từ ConstraintActivator /
        CandidateGenerator bằng attention. FSA chỉ cung cấp API
        allow_free_step_once() cho trường hợp đó.
    """

    target_phrase: str
    target_token_ids: List[int]

    position: int = 0
    state: FSAState = FSAState.IDLE

    # Cho trường hợp đặc biệt theo Hasler:
    # extra token chỉ được phép khi external controller xác nhận
    # attention vẫn nằm trong source span.
    free_before_used: int = 0
    free_after_used: int = 0
    max_free_before: int = 0
    max_free_after: int = 0

    # Optional metadata, không in debug dài.
    meta: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        self.target_phrase = str(
            self.target_phrase or ""
        ).strip()

        self.target_token_ids = [
            int(
                token_id
            )
            for token_id in self.target_token_ids
            if token_id is not None
        ]

        self.position = max(
            0,
            int(
                self.position or 0
            ),
        )

        self.max_free_before = max(
            0,
            int(
                self.max_free_before or 0
            ),
        )

        self.max_free_after = max(
            0,
            int(
                self.max_free_after or 0
            ),
        )

        self.free_before_used = max(
            0,
            int(
                self.free_before_used or 0
            ),
        )

        self.free_after_used = max(
            0,
            int(
                self.free_after_used or 0
            ),
        )

        if not isinstance(
            self.state,
            FSAState,
        ):
            try:
                self.state = FSAState(
                    str(
                        self.state
                    )
                )
            except Exception:
                self.state = FSAState.IDLE

        if self.position >= len(
            self.target_token_ids
        ) and self.target_token_ids:
            self.position = len(
                self.target_token_ids
            )

            if self.state != FSAState.FAILED:
                self.state = FSAState.DONE

    # --------------------------------------------------
    # Basic properties
    # --------------------------------------------------

    @property
    def length(
        self,
    ) -> int:
        return len(
            self.target_token_ids
        )

    @property
    def is_idle(
        self,
    ) -> bool:
        return self.state == FSAState.IDLE

    @property
    def is_forcing(
        self,
    ) -> bool:
        return self.state == FSAState.FORCING

    @property
    def is_done(
        self,
    ) -> bool:
        return self.state == FSAState.DONE

    @property
    def is_failed(
        self,
    ) -> bool:
        return self.state == FSAState.FAILED

    @property
    def is_active(
        self,
    ) -> bool:
        return self.state == FSAState.FORCING

    @property
    def remaining_token_ids(
        self,
    ) -> List[int]:
        if self.is_done or self.is_failed:
            return []

        return list(
            self.target_token_ids[
                self.position:
            ]
        )

    @property
    def consumed_token_ids(
        self,
    ) -> List[int]:
        return list(
            self.target_token_ids[
                : self.position
            ]
        )

    @property
    def progress(
        self,
    ) -> float:
        if self.length == 0:
            return 1.0

        return min(
            1.0,
            max(
                0.0,
                self.position / self.length,
            ),
        )

    # --------------------------------------------------
    # State transitions
    # --------------------------------------------------

    def activate(
        self,
    ) -> None:
        """
        AttentionActivator gọi hàm này khi cross-attention đang
        nhìn vào source span của constraint.

        Nếu target_token_ids rỗng:
            chuyển FAILED vì không có gì để ép.
        """

        if self.is_done or self.is_failed:
            return

        if not self.target_token_ids:
            self.state = FSAState.FAILED
            return

        if self.position >= self.length:
            self.state = FSAState.DONE
            return

        self.state = FSAState.FORCING

    def force_done(
        self,
    ) -> None:
        """
        Chỉ dùng khi hệ thống chắc chắn constraint đã được xử lý đúng.

        Không dùng để vá sau dịch.
        """

        self.position = self.length
        self.state = FSAState.DONE

    def fail(
        self,
    ) -> None:
        self.state = FSAState.FAILED

    def reset(
        self,
    ) -> None:
        self.position = 0
        self.state = FSAState.IDLE
        self.free_before_used = 0
        self.free_after_used = 0

    # --------------------------------------------------
    # Token API for LogitsMasker
    # --------------------------------------------------

    def next_token_id(
        self,
    ) -> Optional[int]:
        """
        Token tiếp theo bắt buộc sinh nếu FSA đang FORCING.
        """

        if not self.is_forcing:
            return None

        if self.position < 0:
            self.position = 0

        if self.position >= self.length:
            self.state = FSAState.DONE
            return None

        return int(
            self.target_token_ids[
                self.position
            ]
        )

    def allowed_token_ids(
        self,
    ) -> List[int]:
        """
        Danh sách token hợp lệ tại bước hiện tại.

        Với FSA chuẩn:
            FORCING -> [next_token_id]
            IDLE/DONE/FAILED -> []
        """

        next_id = self.next_token_id()

        if next_id is None:
            return []

        return [
            int(
                next_id
            )
        ]

    def should_force_token(
        self,
    ) -> bool:
        """
        Cho LogitsMasker biết có cần hard mask không.
        """

        return bool(
            self.allowed_token_ids()
        )

    # --------------------------------------------------
    # Step after token selected
    # --------------------------------------------------

    def step(
        self,
        token_id: int,
    ) -> bool:
        """
        Cập nhật FSA sau khi decoder đã chọn token.

        Return:
            True:
                token hợp lệ.

            False:
                token không hợp lệ với FSA hiện tại.

        Nếu FSA đang IDLE:
            Không nên gọi step().
            Trả True để không làm hỏng hypothesis normal.

        Nếu FSA đang FORCING:
            token_id bắt buộc phải bằng next_token_id().
        """

        token_id = int(
            token_id
        )

        if self.is_done:
            return True

        if self.is_failed:
            return False

        if self.is_idle:
            return True

        expected = self.next_token_id()

        if expected is None:
            if self.position >= self.length:
                self.state = FSAState.DONE
                return True

            self.state = FSAState.FAILED
            return False

        if token_id != expected:
            self.state = FSAState.FAILED
            return False

        self.position += 1

        if self.position >= self.length:
            self.position = self.length
            self.state = FSAState.DONE

        return True

    def step_sequence(
        self,
        token_ids: Sequence[int],
    ) -> bool:
        for token_id in token_ids:
            ok = self.step(
                int(
                    token_id
                )
            )

            if not ok:
                return False

        return True

    # --------------------------------------------------
    # Controlled relaxation API
    # --------------------------------------------------

    def can_use_free_before(
        self,
    ) -> bool:
        """
        Có thể sinh thêm token tự do trước target phrase không?

        Chỉ nên được gọi khi external controller xác nhận:
            attention vẫn nằm trong source span.

        Bản thân FSA không tự quyết định điều này.
        """

        return (
            self.is_idle
            and self.free_before_used < self.max_free_before
        )

    def can_use_free_after(
        self,
    ) -> bool:
        """
        Có thể sinh thêm token tự do sau target phrase không?

        Chỉ dùng cho special cases.
        """

        return (
            self.is_done
            and self.free_after_used < self.max_free_after
        )

    def allow_free_before_once(
        self,
    ) -> bool:
        """
        Ghi nhận một token tự do trước constraint.

        Không activate FSA.
        Sau khi hết quota, activator nên gọi activate().
        """

        if not self.can_use_free_before():
            return False

        self.free_before_used += 1
        return True

    def allow_free_after_once(
        self,
    ) -> bool:
        if not self.can_use_free_after():
            return False

        self.free_after_used += 1
        return True

    # --------------------------------------------------
    # Prefix sync
    # --------------------------------------------------

    def longest_prefix_matched_by_suffix(
        self,
        generated_token_ids: Sequence[int],
    ) -> int:
        """
        Kiểm tra đuôi generated_token_ids đã khớp bao nhiêu token đầu
        của target_token_ids.

        Dùng trong trường hợp model vừa tự sinh một phần target phrase
        đúng lúc attention chạm span, để FSA không ép lại từ đầu.

        Ví dụ:
            target = [bộ, giải, mã]
            generated tail = [bộ]
            -> matched = 1
            FSA tiếp tục ép [giải, mã]
        """

        if not generated_token_ids or not self.target_token_ids:
            return 0

        max_len = min(
            len(
                generated_token_ids
            ),
            len(
                self.target_token_ids
            ),
        )

        best = 0

        generated = [
            int(
                token_id
            )
            for token_id in generated_token_ids
        ]

        target = [
            int(
                token_id
            )
            for token_id in self.target_token_ids
        ]

        for size in range(
            1,
            max_len + 1,
        ):
            if generated[
                -size:
            ] == target[
                :size
            ]:
                best = size

        return best

    def sync_with_generated_tail(
        self,
        generated_token_ids: Sequence[int],
    ) -> int:
        """
        Đồng bộ FSA nếu hypothesis đã tự sinh prefix của target phrase.

        Chỉ dùng ngay lúc activate.
        Không dùng để validate sau dịch.
        """

        if self.is_done or self.is_failed:
            return self.position

        if self.position > 0:
            return self.position

        matched = self.longest_prefix_matched_by_suffix(
            generated_token_ids
        )

        if matched <= 0:
            return self.position

        self.position = matched

        if self.position >= self.length:
            self.state = FSAState.DONE
        else:
            self.state = FSAState.FORCING

        return self.position

    # --------------------------------------------------
    # Cloning
    # --------------------------------------------------

    def clone(
        self,
    ) -> "ConstraintFSA":
        return copy.deepcopy(
            self
        )

    # --------------------------------------------------
    # Compact diagnostics
    # --------------------------------------------------

    def to_dict(
        self,
        compact: bool = True,
    ) -> Dict[str, Any]:
        data = {
            "target_phrase": self.target_phrase,
            "state": self.state.value
            if isinstance(
                self.state,
                FSAState,
            )
            else str(
                self.state
            ),
            "position": self.position,
            "length": self.length,
            "progress": self.progress,
            "next_token_id": self.next_token_id(),
            "is_done": self.is_done,
            "is_failed": self.is_failed,
        }

        if compact:
            return data

        data.update(
            {
                "target_token_ids": list(
                    self.target_token_ids
                ),
                "remaining_token_ids": self.remaining_token_ids,
                "consumed_token_ids": self.consumed_token_ids,
                "free_before_used": self.free_before_used,
                "free_after_used": self.free_after_used,
                "max_free_before": self.max_free_before,
                "max_free_after": self.max_free_after,
                "meta": dict(
                    self.meta
                ),
            }
        )

        return data


# Backward-compatible alias.
# Các file cũ đang import:
#     from constraints.fsa import FSA
# vẫn chạy được.
FSA = ConstraintFSA