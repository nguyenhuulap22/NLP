from __future__ import annotations

import html
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Tuple

import streamlit as st


EXAMPLES = [
    "Fix the bug in the code and deploy the application to the server.",
    "Train the model with a new dataset to improve accuracy.",
    "The API sends a request to the server and receives a JSON response.",
    "The database query is slow because the index is missing.",
    "The transformer uses attention and beam search.",
    "The decoder generates logits for the next token.",
]


# ============================================================
# Session state
# ============================================================

def ensure_session_state() -> None:
    defaults = {
        "history": [],
        "last_result": "",
        "last_terms": [],
        "last_debug": {},
        "last_validation": {},
        "last_trace": [],
        "last_beam_summary": {},
        "input_text": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_input_text() -> None:
    st.session_state.input_text = ""
    st.session_state.last_result = ""
    st.session_state.last_terms = []
    st.session_state.last_debug = {}
    st.session_state.last_validation = {}
    st.session_state.last_trace = []
    st.session_state.last_beam_summary = {}

    if "last_audio" in st.session_state:
        del st.session_state["last_audio"]


def load_sidebar_example(example: str) -> None:
    st.session_state.input_text = example


# ============================================================
# Input handlers
# ============================================================

def handle_audio_input() -> None:
    audio_value = st.session_state.audio_recorder

    if audio_value is None:
        return

    try:
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        audio_value.seek(0)

        with sr.AudioFile(audio_value) as source:
            audio_data = recognizer.record(source)

            text = recognizer.recognize_google(
                audio_data,
                language="en-US",
            )

            if text:
                if st.session_state.input_text:
                    st.session_state.input_text += " " + text
                else:
                    st.session_state.input_text = text

    except ImportError:
        st.session_state.app_error = (
            "Thư viện SpeechRecognition chưa được cài đặt. "
            "Cài bằng: pip install SpeechRecognition"
        )

    except Exception as error:
        st.session_state.app_error = f"Lỗi khi xử lý âm thanh: {error}"


def handle_file_upload() -> None:
    uploaded_file = st.session_state.file_uploader

    if uploaded_file is None:
        return

    try:
        text = read_uploaded_file(uploaded_file)

        if text.strip():
            st.session_state.input_text = text.strip()
            st.session_state.app_success = "Đã đưa nội dung file vào ô dịch."
        else:
            st.session_state.app_warning = "File không có nội dung văn bản."

    except Exception as error:
        st.session_state.app_error = str(error)


def read_uploaded_file(uploaded_file) -> str:
    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if file_name.endswith(
        (
            ".txt",
            ".md",
            ".csv",
            ".json",
            ".py",
            ".html",
            ".css",
            ".js",
        )
    ):
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode(
                "latin-1",
                errors="ignore",
            )

    if file_name.endswith(".docx"):
        try:
            from docx import Document
        except ImportError:
            raise RuntimeError(
                "Bạn cần cài python-docx: pip install python-docx"
            )

        document = Document(BytesIO(file_bytes))

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        return "\n".join(paragraphs)

    if file_name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise RuntimeError(
                "Bạn cần cài pypdf: pip install pypdf"
            )

        reader = PdfReader(BytesIO(file_bytes))
        pages = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n".join(pages)

    raise RuntimeError("Định dạng file chưa được hỗ trợ.")


# ============================================================
# CSS
# ============================================================

def inject_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at 10% 10%, rgba(59,130,246,0.22), transparent 28%),
                    radial-gradient(circle at 90% 15%, rgba(236,72,153,0.20), transparent 30%),
                    linear-gradient(135deg, #eef2ff 0%, #fdf2f8 45%, #ecfeff 100%);
            }

            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 2rem;
                max-width: 1200px;
            }

            .hero-box {
                background: linear-gradient(135deg, #4f46e5, #06b6d4, #ec4899);
                border-radius: 28px;
                padding: 28px 32px;
                color: white !important;
                box-shadow: 0 22px 55px rgba(79,70,229,0.25);
                margin-bottom: 22px;
            }

            .hero-title {
                font-size: 40px;
                font-weight: 900;
                color: white !important;
                margin-bottom: 8px;
            }

            .hero-subtitle {
                color: #f8fafc !important;
                font-size: 16px;
                line-height: 1.7;
                font-weight: 550;
            }

            .result-box {
                background: linear-gradient(145deg, #111827, #1e1b4b, #0f766e);
                color: #f8fafc !important;
                border-radius: 22px;
                padding: 22px;
                min-height: 230px;
                font-size: 17px;
                line-height: 1.8;
                white-space: pre-wrap;
                overflow-wrap: break-word;
                box-shadow: 0 18px 45px rgba(15,23,42,0.25);
            }

            .glossary-chip {
                display: inline-block;
                background: linear-gradient(135deg, #fef3c7, #fde68a);
                border: 1px solid #f59e0b;
                color: #78350f !important;
                border-radius: 999px;
                padding: 7px 12px;
                margin: 5px 6px 5px 0;
                font-size: 13px;
                font-weight: 800;
            }

            .paper-chip {
                display: inline-block;
                background: linear-gradient(135deg, #dbeafe, #cffafe);
                border: 1px solid #06b6d4;
                color: #0f172a !important;
                border-radius: 999px;
                padding: 7px 12px;
                margin: 5px 6px 5px 0;
                font-size: 13px;
                font-weight: 800;
            }

            .paper-chip-ok {
                display: inline-block;
                background: linear-gradient(135deg, #dcfce7, #bbf7d0);
                border: 1px solid #22c55e;
                color: #14532d !important;
                border-radius: 999px;
                padding: 7px 12px;
                margin: 5px 6px 5px 0;
                font-size: 13px;
                font-weight: 800;
            }

            .paper-chip-bad {
                display: inline-block;
                background: linear-gradient(135deg, #fee2e2, #fecaca);
                border: 1px solid #ef4444;
                color: #7f1d1d !important;
                border-radius: 999px;
                padding: 7px 12px;
                margin: 5px 6px 5px 0;
                font-size: 13px;
                font-weight: 800;
            }

            .history-card {
                background: rgba(255,255,255,0.92);
                border-left: 6px solid #6366f1;
                border-radius: 18px;
                padding: 14px 16px;
                margin-bottom: 12px;
                color: #0f172a;
                box-shadow: 0 12px 30px rgba(15,23,42,0.08);
            }

            div[data-testid="stTextArea"] textarea {
                border-radius: 18px !important;
                color: #0f172a !important;
                background: rgba(255,255,255,0.95) !important;
                border: 1.5px solid rgba(99,102,241,0.35) !important;
                font-size: 16px !important;
                line-height: 1.7 !important;
            }

            div[data-testid="stButton"] button {
                border-radius: 14px !important;
                font-weight: 800 !important;
                min-height: 44px;
            }

            div[data-testid="stButton"] button[kind="primary"] {
                background: linear-gradient(135deg, #4f46e5, #06b6d4, #ec4899) !important;
                color: white !important;
                border: none !important;
            }

            div[data-testid="stDownloadButton"] button {
                border-radius: 14px !important;
                font-weight: 800 !important;
                background: linear-gradient(135deg, #10b981, #06b6d4) !important;
                color: white !important;
                border: none !important;
            }

            div[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #eef2ff 0%, #e0f2fe 50%, #fdf2f8 100%);
            }

            footer {
                visibility: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Sidebar / header
# ============================================================

def render_sidebar() -> Tuple[str, str, bool, bool, bool]:
    st.sidebar.title("⚙️ Tùy chọn")

    mode = st.sidebar.radio(
        "Chế độ dịch",
        [
            "Chuẩn CNTT theo bài báo",
            "Dịch nhanh",
            "Dịch từ / ngắn",
        ],
        index=0,
    )

    decoding_mode = st.sidebar.selectbox(
        "Thuật toán giải mã",
        [
            "Multi-stack Beam Search",
            "Greedy Debug",
        ],
        index=0,
    )

    st.sidebar.markdown("#### Mô hình")

    st.sidebar.info(
        "Đang dùng model trong Translator, mặc định: Helsinki-NLP/opus-mt-en-vi"
    )

    show_glossary = st.sidebar.toggle(
        "Hiển thị thuật ngữ nhận diện",
        value=True,
    )

    show_debug = st.sidebar.toggle(
        "Hiển thị debug theo bài báo",
        value=False,
    )

    show_history = st.sidebar.toggle(
        "Hiển thị lịch sử",
        value=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📌 Ví dụ nhanh")

    for i, example in enumerate(EXAMPLES):
        st.sidebar.button(
            example,
            key=f"example_{i}",
            on_click=load_sidebar_example,
            args=(example,),
            use_container_width=True,
        )

    st.sidebar.markdown("---")

    st.sidebar.info(
        "Chế độ Chuẩn CNTT theo bài báo dùng manual constrained decoding: "
        "Attention Monitor, Constraint Activation, FSA, Multi-stack Beam Search "
        "và Covered Span Masking."
    )

    return (
        mode,
        decoding_mode,
        show_glossary,
        show_debug,
        show_history,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">💻 IT Translator Pro</div>
            <div class="hero-subtitle">
                Công cụ dịch Anh → Việt dành cho chuyên ngành Công nghệ thông tin.
                Phiên bản dùng manual constrained decoding theo bài Hasler:
                Attention Monitor, Constraint Activation, FSA, Multi-stack Beam Search
                và Covered Span Masking.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_decoding_value(
    mode: str,
    decoding_mode: str,
) -> str:
    if decoding_mode == "Greedy Debug":
        return "greedy"

    if mode == "Dịch nhanh":
        return "normal"

    if mode == "Dịch từ / ngắn":
        return "greedy"

    return "beam"


# ============================================================
# Result extraction
# ============================================================

def _state_text(
    constraint: Dict[str, Any],
) -> str:
    state = constraint.get("state", "")

    if isinstance(state, dict):
        return str(state.get("value", state))

    return str(state)


def _validation_items_by_id(
    validation: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    details = validation.get("details", []) or validation.get("items", []) or []
    result = {}

    for item in details:
        item_id = item.get("id")

        if item_id is not None:
            result[str(item_id)] = item

    return result


def _display_state_from_validation(
    constraint: Dict[str, Any],
    validation_item: Dict[str, Any] | None,
) -> str:
    raw_state = _state_text(constraint).replace("ConstraintState.", "")

    constraint_type = str(
        constraint.get("constraint_type", "soft")
    ).lower()

    if not validation_item:
        return raw_state

    satisfied = bool(validation_item.get("satisfied", False))
    lexical_found = bool(validation_item.get("lexical_found", False))
    fsa_done = bool(validation_item.get("fsa_done", False))
    state_done = bool(validation_item.get("state_done", False))

    if constraint_type == "soft":
        if lexical_found or satisfied:
            return "FOUND"
        return "PENDING"

    if fsa_done or state_done:
        return "DONE"

    if satisfied:
        return "SATISFIED"

    return raw_state


def extract_terms_from_result(
    result: Dict[str, Any],
) -> List[str]:
    constraints = result.get("constraints_debug", []) or []
    validation = result.get("constraint_validation_debug", {}) or {}
    validation_by_id = _validation_items_by_id(validation)

    terms = []

    for constraint in constraints:
        source = constraint.get("source_phrase")
        target = constraint.get("target_phrase")

        if not source or not target:
            continue

        constraint_id = constraint.get("id")
        validation_item = validation_by_id.get(str(constraint_id))

        constraint_type = str(
            constraint.get("constraint_type", "soft")
        )

        force = bool(
            constraint.get("force", False)
        )

        state = _display_state_from_validation(
            constraint,
            validation_item,
        )

        terms.append(
            f"{source} → {target} [{constraint_type} | {state} | force={force}]"
        )

    return terms


def translate_current_text(
    current_input_text: str,
    mode: str,
    decoding_mode: str,
    translator,
):
    decoding = get_decoding_value(
        mode,
        decoding_mode,
    )

    result = translator.translate(
        current_input_text,
        decoding=decoding,
    )

    terms = extract_terms_from_result(result)

    return result, terms


def add_to_history(
    source: str,
    result: str,
    terms: List[str],
    decoding: str,
) -> None:
    st.session_state.history.insert(
        0,
        {
            "time": datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            "source": source,
            "result": result,
            "terms": terms,
            "decoding": decoding,
        },
    )

    st.session_state.history = st.session_state.history[:10]


# ============================================================
# Render components
# ============================================================

def render_file_upload() -> None:
    st.file_uploader(
        "Tải file cần dịch",
        type=[
            "txt",
            "md",
            "csv",
            "json",
            "py",
            "html",
            "css",
            "js",
            "docx",
            "pdf",
        ],
        key="file_uploader",
        on_change=handle_file_upload,
    )


def render_constraint_chips(
    terms: List[str],
) -> None:
    if terms:
        chips_html = "".join(
            f"<span class='glossary-chip'>{html.escape(term)}</span>"
            for term in terms
        )

        st.markdown(
            chips_html,
            unsafe_allow_html=True,
        )
    else:
        st.info("Không phát hiện thuật ngữ CNTT trong từ điển.")


def render_validation_badges(
    validation: Dict[str, Any],
) -> None:
    if not validation:
        return

    total = int(validation.get("total", 0) or 0)

    satisfied = validation.get("satisfied", None)
    missing = validation.get("missing", None)
    coverage = validation.get("coverage", None)

    if satisfied is None:
        satisfied = validation.get("passed_total", 0)

    if missing is None:
        missing = validation.get("failed_total", 0)

    satisfied = int(satisfied or 0)
    missing = int(missing or 0)

    if coverage is None:
        coverage = satisfied / total if total > 0 else 1.0

    coverage = float(coverage)

    ok = bool(validation.get("ok", False))

    if "ok" not in validation:
        ok = missing == 0

    hard_total = int(
        validation.get(
            "hard_total",
            validation.get("required_total", 0),
        )
        or 0
    )

    hard_satisfied = int(
        validation.get(
            "hard_satisfied",
            validation.get("required_passed", 0),
        )
        or 0
    )

    soft_total = int(validation.get("soft_total", 0) or 0)
    soft_satisfied = int(validation.get("soft_satisfied", 0) or 0)
    fsa_done = int(validation.get("fsa_done", 0) or 0)
    state_done = int(validation.get("state_done", 0) or 0)

    chip_class = "paper-chip-ok" if ok else "paper-chip-bad"
    status = "OK" if ok else "MISSING"

    st.markdown(
        f"""
        <span class='{chip_class}'>
            Constraint status: {status}
        </span>
        <span class='paper-chip'>
            Coverage: {coverage:.2%}
        </span>
        <span class='paper-chip'>
            Satisfied: {satisfied}/{total}
        </span>
        <span class='paper-chip'>
            Missing: {missing}
        </span>
        <span class='paper-chip'>
            Hard: {hard_satisfied}/{hard_total}
        </span>
        <span class='paper-chip'>
            Soft: {soft_satisfied}/{soft_total}
        </span>
        <span class='paper-chip'>
            FSA DONE: {fsa_done}
        </span>
        <span class='paper-chip'>
            State DONE: {state_done}
        </span>
        """,
        unsafe_allow_html=True,
    )

    details = validation.get("details", []) or []
    missing_details = [
        item
        for item in details
        if not bool(
            item.get(
                "satisfied",
                item.get("constraint_satisfied", False),
            )
        )
    ]

    if missing_details:
        with st.expander(
            "⚠️ Thuật ngữ chưa thỏa mãn",
            expanded=False,
        ):
            st.json(missing_details)


def render_paper_debug(
    result: Dict[str, Any],
) -> None:
    st.markdown("#### Constraint Validation")
    st.json(result.get("constraint_validation_debug", {}))

    st.markdown("#### Beam Summary")
    st.json(result.get("beam_summary", {}))

    st.markdown("#### Constraints Debug")
    st.json(result.get("constraints_debug", []))

    st.markdown("#### Decoder Trace")

    trace_debug = result.get("trace_debug", [])

    if trace_debug:
        limit = min(80, len(trace_debug))

        st.caption(
            f"Hiển thị {limit}/{len(trace_debug)} bước trace đầu tiên."
        )

        st.json(trace_debug[:limit])
    else:
        st.info("Chưa có trace để hiển thị.")


def render_history() -> None:
    st.markdown("---")
    st.markdown("### 🕘 Lịch sử dịch")

    if not st.session_state.history:
        st.info("Chưa có lịch sử dịch nào.")
        return

    if st.button(
        "🗑️ Xóa lịch sử",
        use_container_width=True,
    ):
        st.session_state.history = []
        st.rerun()

    for item in st.session_state.history:
        source = html.escape(item.get("source", ""))
        result = html.escape(item.get("result", ""))

        terms_text = html.escape(
            ", ".join(item.get("terms", []))
            if item.get("terms", [])
            else "Không có"
        )

        decoding = html.escape(item.get("decoding", ""))

        st.markdown(
            f"""
            <div class="history-card">
                <b>🕒 {html.escape(item.get("time", ""))}</b><br>
                <b>Giải mã:</b> {decoding}<br>
                <b>Gốc:</b> {source}<br>
                <b>Dịch:</b> {result}<br>
                <b>Thuật ngữ:</b> {terms_text}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# Main app
# ============================================================

def render_app(
    translator,
) -> None:
    ensure_session_state()
    inject_css()

    (
        mode,
        decoding_mode,
        show_glossary,
        show_debug,
        show_history,
    ) = render_sidebar()

    render_header()

    if "app_error" in st.session_state:
        st.error(st.session_state.app_error)
        del st.session_state.app_error

    if "app_warning" in st.session_state:
        st.warning(st.session_state.app_warning)
        del st.session_state.app_warning

    if "app_success" in st.session_state:
        st.success(st.session_state.app_success)
        del st.session_state.app_success

    left, right = st.columns(
        [
            1.05,
            0.95,
        ],
        gap="large",
    )

    translate_clicked = False

    with left:
        st.subheader("Văn bản tiếng Anh")

        tab_text, tab_file, tab_audio = st.tabs(
            [
                "✍️ Nhập văn bản",
                "📄 Tải file",
                "🎤 Giọng nói",
            ]
        )

        with tab_text:
            st.caption("Nhập hoặc dán văn bản trực tiếp vào ô bên dưới.")

        with tab_file:
            render_file_upload()

        with tab_audio:
            st.audio_input(
                "Nhấn vào biểu tượng Micro để nói",
                key="audio_recorder",
                on_change=handle_audio_input,
            )

        st.text_area(
            "Nhập nội dung cần dịch",
            key="input_text",
            height=250,
            placeholder=(
                "Ví dụ: Fix the bug in the code and deploy "
                "the application to the server."
            ),
            label_visibility="collapsed",
        )

        col_1, col_2 = st.columns(2)

        with col_1:
            translate_clicked = st.button(
                "Dịch ngay",
                type="primary",
                use_container_width=True,
            )

        with col_2:
            st.button(
                "Xóa nội dung",
                use_container_width=True,
                on_click=clear_input_text,
            )

    with right:
        st.subheader("Kết quả tiếng Việt")

        tab_result = st.tabs(["💡 Bản dịch"])

        with tab_result[0]:
            current_input = st.session_state.get(
                "input_text",
                "",
            ).strip()

            auto_translate = st.session_state.pop(
                "auto_translate",
                False,
            )

            if translate_clicked or auto_translate:
                if not current_input:
                    st.warning("Bạn chưa nhập văn bản để dịch.")
                else:
                    with st.spinner(
                        "Đang chạy manual constrained decoding: Attention → FSA → Multi-stack Beam Search..."
                    ):
                        result, terms = translate_current_text(
                            current_input,
                            mode,
                            decoding_mode,
                            translator,
                        )

                        st.session_state.last_result = result.get(
                            "translation",
                            "",
                        )

                        st.session_state.last_terms = terms
                        st.session_state.last_debug = result

                        st.session_state.last_validation = result.get(
                            "constraint_validation_debug",
                            {},
                        )

                        st.session_state.last_trace = result.get(
                            "trace_debug",
                            [],
                        )

                        st.session_state.last_beam_summary = result.get(
                            "beam_summary",
                            {},
                        )

                        add_to_history(
                            current_input,
                            st.session_state.last_result,
                            terms,
                            result.get("decoding", ""),
                        )

            current_result = st.session_state.last_result
            current_terms = st.session_state.last_terms

            if current_result:
                safe_result = html.escape(current_result)

                st.markdown(
                    f"<div class='result-box'>{safe_result}</div>",
                    unsafe_allow_html=True,
                )

                st.download_button(
                    "⬇️ Tải bản dịch (.txt)",
                    data=current_result.encode("utf-8"),
                    file_name="ban_dich_cntt.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

                if show_glossary:
                    st.markdown("#### Thuật ngữ nhận diện")
                    render_constraint_chips(current_terms)

                render_validation_badges(
                    st.session_state.last_validation
                )

            else:
                st.markdown(
                    "<div class='result-box'>Bản dịch sẽ hiển thị ở đây.</div>",
                    unsafe_allow_html=True,
                )

    if show_debug and st.session_state.last_debug:
        st.markdown("---")
        st.markdown("### 🔬 Debug theo bài báo")
        render_paper_debug(st.session_state.last_debug)

    if show_history:
        render_history()

    with st.expander("📘 Hướng dẫn sử dụng"):
        st.markdown(
            """
            1. Nhập văn bản tiếng Anh hoặc tải file lên.  
            2. Chọn **Chuẩn CNTT theo bài báo** để dùng manual constrained decoding.  
            3. Bấm **Dịch ngay**.  
            4. Xem phần **Thuật ngữ nhận diện** để biết constraint nào được phát hiện.  
            5. Bật **Hiển thị debug theo bài báo** để xem Attention, Constraint Activation, FSA, Multi-stack Beam Search và Constraint Validation.  
            """
        )

    st.markdown("---")

    st.caption(
        "Made with Python + Streamlit + MarianMT | Constrained NMT: Attention + Constraint Activation + FSA + Multi-stack Beam Search"
    )