import html
import threading
from datetime import datetime
from io import BytesIO

import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

from dictionary import (
    translate_it_text,
    find_glossary_terms,
    translate_with_model,
    postprocess_vi,
    normalize_text,
    translate_word_by_word,
)


EXAMPLES = [
    "Fix the bug in the code and deploy the application to the server.",
    "Train the model with a new dataset to improve accuracy.",
]


VOICE_LOCK = threading.Lock()

VOICE_STATE = {
    "listening": False,
    "text": "",
    "error": "",
    "stopper": None,
}


def ensure_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []

    if "last_result" not in st.session_state:
        st.session_state.last_result = ""

    if "last_terms" not in st.session_state:
        st.session_state.last_terms = []

    if "input_text" not in st.session_state:
        st.session_state.input_text = ""


def clear_input_text() -> None:
    st.session_state["input_text"] = ""
    st.session_state.last_result = ""
    st.session_state.last_terms = []


def load_sidebar_example(example: str) -> None:
    st.session_state["input_text"] = example


def read_uploaded_file(uploaded_file) -> str:
    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if file_name.endswith((".txt", ".md", ".csv", ".json", ".py", ".html", ".css", ".js")):
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1", errors="ignore")

    if file_name.endswith(".docx"):
        try:
            from docx import Document
        except ImportError:
            raise RuntimeError("Bạn cần cài python-docx bằng lệnh: pip install python-docx")

        document = Document(BytesIO(file_bytes))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    if file_name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise RuntimeError("Bạn cần cài pypdf bằng lệnh: pip install pypdf")

        reader = PdfReader(BytesIO(file_bytes))
        pages = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)

        return "\n".join(pages)

    raise RuntimeError("Định dạng file chưa được hỗ trợ. Hãy dùng .txt, .docx hoặc .pdf.")


def start_realtime_voice_input() -> None:
    try:
        import speech_recognition as sr
    except ImportError:
        with VOICE_LOCK:
            VOICE_STATE["error"] = "Bạn cần cài SpeechRecognition bằng lệnh: pip install SpeechRecognition"
        return

    with VOICE_LOCK:
        if VOICE_STATE["listening"]:
            return

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8

    try:
        microphone = sr.Microphone()
    except Exception:
        with VOICE_LOCK:
            VOICE_STATE["error"] = (
                "Không mở được microphone. Nếu dùng Windows, hãy cài PyAudio bằng lệnh: "
                "pip install pipwin rồi chạy tiếp: pipwin install pyaudio"
            )
        return

    def callback(recognizer_obj, audio_data):
        try:
            text = recognizer_obj.recognize_google(audio_data, language="en-US")

            if text.strip():
                with VOICE_LOCK:
                    old_text = VOICE_STATE["text"].strip()

                    if old_text:
                        VOICE_STATE["text"] = old_text + " " + text.strip()
                    else:
                        VOICE_STATE["text"] = text.strip()

                    VOICE_STATE["error"] = ""

        except sr.UnknownValueError:
            pass

        except sr.RequestError:
            with VOICE_LOCK:
                VOICE_STATE["error"] = (
                    "Không kết nối được dịch vụ nhận diện giọng nói. "
                    "Hãy kiểm tra Internet."
                )

        except Exception as error:
            with VOICE_LOCK:
                VOICE_STATE["error"] = f"Lỗi nhận diện giọng nói: {error}"

    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.6)

        stopper = recognizer.listen_in_background(
            microphone,
            callback,
            phrase_time_limit=4,
        )

        with VOICE_LOCK:
            VOICE_STATE["listening"] = True
            VOICE_STATE["stopper"] = stopper
            VOICE_STATE["error"] = ""

    except Exception as error:
        with VOICE_LOCK:
            VOICE_STATE["error"] = f"Không thể bắt đầu nghe giọng nói: {error}"


def stop_realtime_voice_input() -> None:
    with VOICE_LOCK:
        stopper = VOICE_STATE.get("stopper")

    if stopper:
        try:
            stopper(wait_for_stop=False)
        except Exception:
            pass

    with VOICE_LOCK:
        VOICE_STATE["listening"] = False
        VOICE_STATE["stopper"] = None


def clear_voice_text() -> None:
    with VOICE_LOCK:
        VOICE_STATE["text"] = ""
        VOICE_STATE["error"] = ""

    st.session_state["input_text"] = ""


def inject_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at 10% 10%, rgba(59, 130, 246, 0.28), transparent 28%),
                    radial-gradient(circle at 90% 15%, rgba(236, 72, 153, 0.24), transparent 30%),
                    radial-gradient(circle at 50% 95%, rgba(34, 197, 94, 0.18), transparent 32%),
                    linear-gradient(135deg, #eef2ff 0%, #fdf2f8 45%, #ecfeff 100%);
            }

            .main {
                background: transparent;
            }

            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 2.2rem;
                max-width: 1220px;
            }

            .hero-box {
                background:
                    linear-gradient(135deg, rgba(79, 70, 229, 0.96), rgba(14, 165, 233, 0.92), rgba(236, 72, 153, 0.90));
                border: 1px solid rgba(255, 255, 255, 0.35);
                border-radius: 30px;
                padding: 30px 32px 24px 32px;
                box-shadow:
                    0 24px 60px rgba(79, 70, 229, 0.25),
                    inset 0 1px 0 rgba(255, 255, 255, 0.35);
                margin-bottom: 22px;
                position: relative;
                overflow: hidden;
                transition: all 0.28s ease !important;
            }

            .hero-box::before {
                content: "";
                position: absolute;
                width: 220px;
                height: 220px;
                right: -70px;
                top: -70px;
                background: rgba(255,255,255,0.18);
                border-radius: 50%;
            }

            .hero-box::after {
                content: "";
                position: absolute;
                width: 150px;
                height: 150px;
                left: -50px;
                bottom: -50px;
                background: rgba(255,255,255,0.14);
                border-radius: 50%;
            }

            .hero-box:hover {
                transform: translateY(-4px);
                box-shadow:
                    0 30px 75px rgba(79, 70, 229, 0.32),
                    inset 0 1px 0 rgba(255, 255, 255, 0.42);
            }

            .hero-title {
                font-size: 40px;
                font-weight: 900;
                color: #ffffff !important;
                margin-bottom: 8px;
                letter-spacing: -0.8px;
                position: relative;
                z-index: 1;
                text-shadow: 0 3px 12px rgba(15, 23, 42, 0.22);
            }

            .hero-subtitle {
                font-size: 16px;
                color: #f8fafc !important;
                line-height: 1.75;
                font-weight: 550;
                max-width: 900px;
                position: relative;
                z-index: 1;
            }

            .metric-card {
                background:
                    linear-gradient(145deg, rgba(255,255,255,0.92), rgba(248,250,252,0.86));
                border: 1px solid rgba(255,255,255,0.75);
                border-radius: 24px;
                padding: 22px 18px;
                text-align: center;
                color: #0f172a;
                box-shadow:
                    0 16px 38px rgba(15, 23, 42, 0.11),
                    inset 0 1px 0 rgba(255,255,255,0.85);
                min-height: 118px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                backdrop-filter: blur(12px);
                transition: all 0.28s ease !important;
            }

            .metric-card:hover {
                transform: translateY(-6px) scale(1.015);
                border-color: rgba(99, 102, 241, 0.65);
                box-shadow:
                    0 24px 55px rgba(79, 70, 229, 0.22),
                    0 0 0 4px rgba(99, 102, 241, 0.08),
                    inset 0 1px 0 rgba(255,255,255,0.95);
            }

            .metric-card h4 {
                color: #312e81 !important;
                margin: 0 0 10px 0;
                font-size: 23px;
                font-weight: 900;
            }

            .metric-card p {
                color: #334155 !important;
                margin: 0;
                font-size: 16px;
                font-weight: 700;
                line-height: 1.5;
            }

            .input-tool-box {
                background:
                    linear-gradient(145deg, rgba(255,255,255,0.92), rgba(241,245,249,0.84));
                border: 1px solid rgba(255,255,255,0.70);
                border-radius: 24px;
                padding: 18px;
                margin-bottom: 16px;
                box-shadow: 0 15px 38px rgba(15, 23, 42, 0.08);
                backdrop-filter: blur(12px);
                transition: all 0.28s ease !important;
            }

            .input-tool-box:hover {
                transform: translateY(-3px);
                border-color: rgba(14, 165, 233, 0.65);
                box-shadow:
                    0 20px 48px rgba(14, 165, 233, 0.14),
                    0 0 0 4px rgba(14, 165, 233, 0.08);
            }

            .input-tool-title {
                font-size: 17px;
                font-weight: 900;
                color: #1e1b4b !important;
                margin-bottom: 10px;
            }

            .input-tool-note {
                font-size: 13.5px;
                color: #475569 !important;
                line-height: 1.6;
                margin-top: 8px;
                font-weight: 550;
            }

            .result-box {
                background:
                    linear-gradient(145deg, #111827 0%, #1e1b4b 48%, #0f766e 100%);
                border: 1px solid rgba(125, 211, 252, 0.35);
                border-radius: 24px;
                padding: 22px;
                color: #f8fafc !important;
                line-height: 1.85;
                min-height: 220px;
                font-size: 17px;
                font-weight: 550;
                white-space: pre-wrap;
                overflow-wrap: break-word;
                box-shadow:
                    0 18px 48px rgba(15, 23, 42, 0.22),
                    inset 0 1px 0 rgba(255,255,255,0.12);
                transition: all 0.28s ease !important;
            }

            .result-box:hover {
                transform: translateY(-3px);
                border-color: rgba(125, 211, 252, 0.65);
                box-shadow:
                    0 24px 60px rgba(15, 23, 42, 0.30),
                    0 0 0 4px rgba(14, 165, 233, 0.10),
                    inset 0 1px 0 rgba(255,255,255,0.16);
            }

            .result-box * {
                color: #f8fafc !important;
            }

            .glossary-chip {
                display: inline-block;
                background: linear-gradient(135deg, #fef3c7, #fde68a);
                border: 1px solid #f59e0b;
                color: #78350f !important;
                border-radius: 999px;
                padding: 8px 13px;
                margin: 5px 6px 5px 0;
                font-size: 13px;
                font-weight: 800;
                box-shadow: 0 8px 18px rgba(245, 158, 11, 0.16);
                transition: all 0.28s ease !important;
            }

            .glossary-chip:hover {
                transform: translateY(-3px) scale(1.05);
                background: linear-gradient(135deg, #fde68a, #fbbf24);
                box-shadow: 0 12px 24px rgba(245, 158, 11, 0.28);
                cursor: default;
            }

            .small-note {
                color: #334155 !important;
                font-size: 13px;
                line-height: 1.6;
                font-weight: 550;
            }

            .history-card {
                background:
                    linear-gradient(145deg, rgba(255,255,255,0.94), rgba(248,250,252,0.88));
                border: 1px solid rgba(255,255,255,0.75);
                border-left: 6px solid #6366f1;
                border-radius: 20px;
                padding: 15px 16px;
                margin-bottom: 12px;
                color: #0f172a;
                box-shadow: 0 14px 35px rgba(15, 23, 42, 0.075);
                backdrop-filter: blur(10px);
                transition: all 0.28s ease !important;
            }

            .history-card:hover {
                transform: translateX(6px);
                border-left-color: #ec4899;
                box-shadow:
                    0 18px 45px rgba(236, 72, 153, 0.14),
                    0 0 0 4px rgba(236, 72, 153, 0.06);
            }

            h1, h2, h3, h4, h5, h6, p, label, div, span {
                color: #0f172a;
            }

            h3 {
                font-weight: 900 !important;
                color: #1e1b4b !important;
            }

            div[data-testid="stTextArea"] textarea {
                border-radius: 22px !important;
                background: rgba(255, 255, 255, 0.92) !important;
                color: #0f172a !important;
                border: 1.5px solid rgba(99, 102, 241, 0.35) !important;
                font-size: 16px !important;
                font-weight: 550 !important;
                line-height: 1.75 !important;
                box-shadow:
                    0 12px 28px rgba(15, 23, 42, 0.06),
                    inset 0 1px 0 rgba(255,255,255,0.85);
                transition: all 0.22s ease !important;
            }

            div[data-testid="stTextArea"] textarea:hover {
                border-color: #38bdf8 !important;
                box-shadow:
                    0 16px 34px rgba(14, 165, 233, 0.13),
                    0 0 0 4px rgba(14, 165, 233, 0.08) !important;
            }

            div[data-testid="stTextArea"] textarea:focus {
                border-color: #6366f1 !important;
                box-shadow:
                    0 0 0 4px rgba(99, 102, 241, 0.16),
                    0 14px 30px rgba(99, 102, 241, 0.10) !important;
            }

            div[data-testid="stTextArea"] textarea::placeholder {
                color: #64748b !important;
                opacity: 1 !important;
            }

            div[data-testid="stButton"] button {
                border-radius: 16px !important;
                font-weight: 850 !important;
                min-height: 46px;
                border: 1px solid rgba(255,255,255,0.55) !important;
                box-shadow: 0 12px 24px rgba(15, 23, 42, 0.10);
                transition: all 0.22s ease !important;
            }

            div[data-testid="stButton"] button:not(:disabled):hover {
                transform: translateY(-3px) scale(1.02);
                filter: brightness(1.04);
                box-shadow:
                    0 18px 36px rgba(79, 70, 229, 0.25),
                    0 0 0 4px rgba(99, 102, 241, 0.10);
            }

            div[data-testid="stButton"] button:not(:disabled):active {
                transform: translateY(0px) scale(0.98);
            }

            div[data-testid="stButton"] button[kind="primary"] {
                background: linear-gradient(135deg, #4f46e5, #06b6d4, #ec4899) !important;
                border: none !important;
                color: white !important;
            }

            div[data-testid="stDownloadButton"] button {
                border-radius: 16px !important;
                font-weight: 850 !important;
                min-height: 46px;
                background: linear-gradient(135deg, #10b981, #06b6d4) !important;
                color: white !important;
                border: none !important;
                box-shadow: 0 12px 26px rgba(16, 185, 129, 0.22);
                transition: all 0.22s ease !important;
            }

            div[data-testid="stDownloadButton"] button:not(:disabled):hover {
                transform: translateY(-3px) scale(1.02);
                filter: brightness(1.06);
                box-shadow:
                    0 18px 36px rgba(16, 185, 129, 0.28),
                    0 0 0 4px rgba(16, 185, 129, 0.10);
            }

            div[data-testid="stFileUploader"] {
                background: rgba(255,255,255,0.60);
                border-radius: 18px;
                padding: 10px;
                border: 1px dashed rgba(99, 102, 241, 0.45);
                transition: all 0.25s ease;
            }

            div[data-testid="stFileUploader"]:hover {
                border-color: #ec4899 !important;
                background: rgba(255, 255, 255, 0.78);
                box-shadow:
                    0 16px 34px rgba(236, 72, 153, 0.13),
                    0 0 0 4px rgba(236, 72, 153, 0.07);
            }

            div[data-testid="stTabs"] button {
                font-weight: 850 !important;
                color: #334155 !important;
                transition: all 0.22s ease !important;
                border-radius: 12px 12px 0 0 !important;
            }

            div[data-testid="stTabs"] button:hover {
                background: rgba(99, 102, 241, 0.10) !important;
                color: #4f46e5 !important;
                transform: translateY(-2px);
            }

            div[data-testid="stTabs"] button[aria-selected="true"] {
                color: #4f46e5 !important;
            }

            div[data-testid="stSidebar"] {
                background:
                    linear-gradient(180deg, #eef2ff 0%, #e0f2fe 48%, #fdf2f8 100%);
                border-right: 1px solid rgba(148,163,184,0.45);
            }

            div[data-testid="stSidebar"] * {
                color: #0f172a !important;
            }

            section[data-testid="stSidebar"] button:hover {
                transform: translateX(4px);
                box-shadow: 0 12px 26px rgba(99, 102, 241, 0.18);
            }

            .stRadio label,
            .stCheckbox label,
            .stToggle label {
                font-weight: 700 !important;
            }

            .stAlert {
                border-radius: 18px !important;
                border: 1px solid rgba(255,255,255,0.5) !important;
                box-shadow: 0 10px 25px rgba(15,23,42,0.06);
            }

            .stCodeBlock {
                border-radius: 18px !important;
            }

            div[data-testid="stExpander"] {
                transition: all 0.25s ease !important;
            }

            div[data-testid="stExpander"]:hover {
                transform: translateY(-2px);
                filter: brightness(1.02);
            }

            hr {
                border: none;
                height: 1px;
                background: linear-gradient(90deg, transparent, rgba(99,102,241,0.45), transparent);
                margin-top: 28px;
                margin-bottom: 16px;
            }

            footer {
                visibility: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[str, bool, bool]:
    st.sidebar.title("⚙️ Tùy chọn")

    mode = st.sidebar.radio(
        "Chế độ dịch",
        ["Chuẩn CNTT", "Dịch nhanh", "Dịch từ / ngắn"],
        index=0,
        help="Chuẩn CNTT ưu tiên ngữ cảnh kỹ thuật, Dịch từ / ngắn dịch từng từ hoặc cụm ngắn không dựa vào ngữ cảnh.",
    )

    show_glossary = st.sidebar.toggle("Hiển thị thuật ngữ nhận diện", value=True)
    show_history = st.sidebar.toggle("Hiển thị lịch sử", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📌 Ví dụ nhanh")

    for i, ex in enumerate(EXAMPLES):
        st.sidebar.button(
            ex,
            key=f"example_{i}",
            on_click=load_sidebar_example,
            args=(ex,),
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div class='small-note'>
            Ứng dụng dùng mô hình dịch Anh → Việt kết hợp từ điển chuyên ngành CNTT
            và luật ngữ cảnh để giảm lỗi dịch sai thuật ngữ.
        </div>
        """,
        unsafe_allow_html=True,
    )

    return mode, show_glossary, show_history


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">💻 IT Translator Pro</div>
            <div class="hero-subtitle">
                Công cụ dịch thuật Anh → Việt dành cho chuyên ngành CNTT.
                Hệ thống kết hợp mô hình dịch tự động, từ điển thuật ngữ công nghệ thông tin
                và luật ngữ cảnh để ưu tiên bản dịch sát nghĩa hơn cho tài liệu kỹ thuật.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    spacer_left, col_m1, col_m2, spacer_right = st.columns([0.25, 1, 1, 0.25])

    with col_m1:
        st.markdown(
            """
            <div class='metric-card'>
                <h4>Hỗ trợ</h4>
                <p>Anh → Việt</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m2:
        st.markdown(
            """
            <div class='metric-card'>
                <h4>Chuyên ngành</h4>
                <p>CNTT / lập trình / AI / mạng</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def add_to_history(source: str, result: str, terms: list[str]) -> None:
    st.session_state.history.insert(
        0,
        {
            "time": datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            "source": source,
            "result": result,
            "terms": terms,
        },
    )

    st.session_state.history = st.session_state.history[:10]


def translate_current_text(current_input_text: str, mode: str, tokenizer, model):
    if mode == "Chuẩn CNTT":
        result, terms = translate_it_text(current_input_text, tokenizer, model)

    elif mode == "Dịch nhanh":
        raw = translate_with_model(normalize_text(current_input_text), tokenizer, model)
        result = postprocess_vi(raw, source_text=current_input_text)
        terms = find_glossary_terms(current_input_text)

    else:
        result = translate_word_by_word(current_input_text)
        terms = find_glossary_terms(current_input_text)

    return result, terms


def handle_file_upload_before_text_area() -> None:
    st.markdown("<div class='input-tool-box'>", unsafe_allow_html=True)
    st.markdown("<div class='input-tool-title'>📄 Tải file lên</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Chọn file cần đưa vào ô dịch",
        type=["txt", "md", "csv", "json", "py", "html", "css", "js", "docx", "pdf"],
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <div class='input-tool-note'>
            Hỗ trợ TXT, DOCX, PDF, Markdown, CSV, JSON và một số file code.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if uploaded_file is not None:
        file_signature = f"{uploaded_file.name}_{uploaded_file.size}"

        if st.session_state.get("last_uploaded_file") != file_signature:
            try:
                file_text = read_uploaded_file(uploaded_file)

                if file_text.strip():
                    st.session_state["input_text"] = file_text.strip()
                    st.session_state["last_uploaded_file"] = file_signature
                    st.success("Đã tải nội dung file vào ô nhập.")
                else:
                    st.warning("File không có nội dung văn bản để đọc.")

            except RuntimeError as error:
                st.error(str(error))

    st.markdown("</div>", unsafe_allow_html=True)


def handle_voice_input_before_text_area() -> bool:
    st.markdown("<div class='input-tool-box'>", unsafe_allow_html=True)
    st.markdown("<div class='input-tool-title'>🎙️ Nhập bằng giọng nói trực tiếp</div>", unsafe_allow_html=True)

    with VOICE_LOCK:
        is_listening = VOICE_STATE["listening"]
        live_text = VOICE_STATE["text"]
        error_text = VOICE_STATE["error"]

    if is_listening and st_autorefresh is not None:
        st_autorefresh(interval=1200, key="voice_live_refresh")

    if is_listening and st_autorefresh is None:
        st.warning(
            "Bạn cần cài streamlit-autorefresh để chữ tự cập nhật liên tục: "
            "pip install streamlit-autorefresh"
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.button(
            "Bắt đầu nói",
            use_container_width=True,
            disabled=is_listening,
            on_click=start_realtime_voice_input,
        )

    with c2:
        st.button(
            "Dừng ghi âm",
            use_container_width=True,
            disabled=not is_listening,
            on_click=stop_realtime_voice_input,
        )

    with c3:
        st.button(
            "Xóa giọng nói",
            use_container_width=True,
            on_click=clear_voice_text,
        )

    if is_listening:
        st.success("Đang nghe... bạn cứ nói tiếng Anh, văn bản sẽ hiện dần bên dưới.")
    else:
        st.info("Bấm **Bắt đầu nói** rồi nói tiếng Anh. Khi xong thì bấm **Dừng ghi âm**.")

    if error_text:
        st.error(error_text)

    st.text_area(
        "Văn bản nhận diện trực tiếp",
        value=live_text,
        height=160,
        disabled=True,
    )

    if live_text.strip():
        st.session_state["input_text"] = live_text.strip()

    translate_voice_clicked = st.button(
        "Dịch văn bản giọng nói",
        use_container_width=True,
        type="primary",
        disabled=not live_text.strip(),
    )

    st.markdown(
        """
        <div class='input-tool-note'>
            Văn bản sẽ cập nhật theo từng cụm sau khi bạn ngắt hơi ngắn.
            Khi văn bản tiếng Anh hiện ra, bấm <b>Dịch văn bản giọng nói</b> để dịch sang tiếng Việt.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    return translate_voice_clicked


def render_app(tokenizer, model) -> None:
    st.set_page_config(
        page_title="IT Translator Pro",
        page_icon="💻",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    ensure_session_state()
    inject_css()

    mode, show_glossary, show_history = render_sidebar()

    render_header()

    left, right = st.columns([1.08, 0.92], gap="large")

    translate_clicked = False
    voice_translate_clicked = False

    with left:
        st.subheader("Văn bản tiếng Anh")

        tab_file, tab_voice, tab_text = st.tabs(
            ["📄 Tải file", "🎙️ Giọng nói", "✍️ Nhập văn bản"]
        )

        with tab_file:
            handle_file_upload_before_text_area()

        with tab_voice:
            voice_translate_clicked = handle_voice_input_before_text_area()

        with tab_text:
            input_text = st.text_area(
                "Nhập nội dung cần dịch",
                key="input_text",
                height=260,
                placeholder="Ví dụ: Train the model with a new dataset and deploy the application to the server.",
                label_visibility="collapsed",
            )

            c1, c2 = st.columns(2)

            with c1:
                translate_clicked = st.button(
                    "Dịch ngay",
                    use_container_width=True,
                    type="primary",
                )

            with c2:
                st.button(
                    "Xóa nội dung",
                    use_container_width=True,
                    on_click=clear_input_text,
                )

    with right:
        st.subheader("Kết quả tiếng Việt")

        result_placeholder = st.empty()
        terms_placeholder = st.empty()

        current_input_text = st.session_state.get("input_text", "").strip()

        if translate_clicked or voice_translate_clicked:
            if not current_input_text:
                st.warning("Bạn chưa nhập văn bản để dịch.")
            else:
                with st.spinner("Đang phân tích thuật ngữ và dịch văn bản..."):
                    result, terms = translate_current_text(
                        current_input_text,
                        mode,
                        tokenizer,
                        model,
                    )

                    st.session_state.last_result = result
                    st.session_state.last_terms = terms

                    add_to_history(current_input_text, result, terms)

        current_result = st.session_state.last_result
        current_terms = st.session_state.last_terms

        if current_result:
            safe_result = html.escape(current_result)

            result_placeholder.markdown(
                f"<div class='result-box'>{safe_result}</div>",
                unsafe_allow_html=True,
            )

            if show_glossary:
                if current_terms:
                    chips_html = "".join(
                        [
                            f"<span class='glossary-chip'>{html.escape(term)}</span>"
                            for term in current_terms
                        ]
                    )
                    terms_placeholder.markdown(chips_html, unsafe_allow_html=True)
                else:
                    terms_placeholder.info(
                        "Không phát hiện thuật ngữ CNTT nổi bật trong đoạn này."
                    )

            col_a, col_b = st.columns(2)

            with col_a:
                st.download_button(
                    "⬇️ Tải bản dịch (.txt)",
                    data=current_result.encode("utf-8"),
                    file_name="ban_dich_cntt.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            with col_b:
                st.code(current_result, language=None)

        else:
            result_placeholder.markdown(
                "<div class='result-box'>Bản dịch.</div>",
                unsafe_allow_html=True,
            )

    if show_history:
        st.markdown("### 🕘 Lịch sử dịch")

        if st.session_state.history:
            if st.button("🗑️ Xóa lịch sử", use_container_width=True):
                st.session_state.history = []
                st.rerun()

            for item in st.session_state.history:
                source = html.escape(item["source"])
                result = html.escape(item["result"])
                terms_text = html.escape(
                    ", ".join(item["terms"]) if item["terms"] else "Không có"
                )

                st.markdown(
                    f"""
                    <div class="history-card">
                        <b>🕒 {item['time']}</b><br>
                        <b>Gốc:</b> {source}<br>
                        <b>Dịch:</b> {result}<br>
                        <b>Thuật ngữ:</b> {terms_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("Chưa có lịch sử dịch nào.")

    with st.expander("📘 Hướng dẫn sử dụng"):
        st.markdown(
            """
            1. Có thể tải file, nói bằng giọng nói hoặc nhập văn bản thủ công.  
            2. Nếu dùng giọng nói, bấm **Bắt đầu nói**, sau đó nói tiếng Anh.  
            3. Văn bản nhận diện sẽ hiện dần.  
            4. Bấm **Dịch văn bản giọng nói** để dịch nội dung vừa nói.  
            5. Chọn **Chuẩn CNTT** nếu văn bản có thuật ngữ kỹ thuật.
            """
        )

    with st.expander("ℹ️ Lưu ý về độ chính xác"):
        st.markdown(
            """
            Ứng dụng này được tối ưu cho văn bản CNTT bằng cách kết hợp:

            - mô hình dịch tự động Anh → Việt,
            - từ điển thuật ngữ CNTT,
            - luật ngữ cảnh cho các từ dễ dịch sai.

            Tính năng giọng nói cần Internet vì đang dùng dịch vụ nhận diện giọng nói trực tuyến.
            Văn bản giọng nói sẽ cập nhật theo từng cụm ngắn, không phải từng chữ tuyệt đối.
            """
        )

    st.markdown("---")
    st.caption(
        "Made with Python + Streamlit + MarianMT | Tối ưu cho tài liệu chuyên ngành CNTT"
    )