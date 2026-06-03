"""
English → Vietnamese IT Translation Pipeline
=============================================
Refactored for accuracy, stability, and maintainability.

Pipeline order:
 1. Mask special content (URLs, code blocks, numbers).
 2. Normalize text (ensure space after punctuation).
 3. Chunk unpunctuated text into manageable pieces.
 4. Split into sentences.
 Per sentence:
   5. Mask glossary phrases → placeholder tokens.
   6. Mask high-risk single terms → placeholder tokens.
   7. Run model on masked input.
   8. Denoise: Fix model-mangled mask tokens (e.g., XPHRIRS2X → XPHRASE2X).
   9. Corruption guard → fallback to dictionary if bad output.
   10. Unmask terms → insert correct Vietnamese for single terms.
   11. Unmask phrases → insert correct Vietnamese for phrases.
   12. Domain overlay (only fix incorrect translations).
   13. Local postprocess.
 14. Merge and global postprocess (with source context).
 15. Unmask special content.
"""

import re
from typing import Dict, List, Tuple, Optional

import streamlit as st
from transformers import MarianMTModel, MarianTokenizer


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model() -> Tuple[MarianTokenizer, MarianMTModel]:
    model_name = "Helsinki-NLP/opus-mt-en-vi"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model


# ============================================================
# GLOSSARY DATA
# ============================================================

# Priority 1 – Multi-word phrases (highest priority)
PHRASE_GLOSSARY: Dict[str, str] = {
    "machine learning": "học máy",
    "deep learning": "học sâu",
    "neural network": "mạng nơ-ron",
    "artificial intelligence": "trí tuệ nhân tạo",
    "source code": "mã nguồn",
    "user interface": "giao diện người dùng",
    "virtual machine": "máy ảo",
    "operating system": "hệ điều hành",
    "database query": "truy vấn cơ sở dữ liệu",
    "training data": "dữ liệu huấn luyện",
    "training dataset": "tập dữ liệu huấn luyện",
    "web server": "máy chủ web",
    "application server": "máy chủ ứng dụng",
    "cloud computing infrastructure": "cơ sở hạ tầng điện toán đám mây",
    "cloud computing": "điện toán đám mây",
    "computer network": "mạng máy tính",
    "data structure": "cấu trúc dữ liệu",
    "machine code": "mã máy",
    "programming language": "ngôn ngữ lập trình",
    "object oriented programming": "lập trình hướng đối tượng",
    "sql query": "truy vấn SQL",
    "primary key": "khóa chính",
    "foreign key": "khóa ngoại",
    "runtime environment": "môi trường chạy",
    "background process": "tiến trình nền",
    "application programming interface": "giao diện lập trình ứng dụng",
    "high traffic": "lưu lượng cao",
    "network traffic": "lưu lượng mạng",
    "load balancer": "bộ cân bằng tải",
    "load balancing": "cân bằng tải",
    "system performance": "hiệu năng hệ thống",
    "access control": "kiểm soát truy cập",
    "version control": "quản lý phiên bản",
    "open source": "mã nguồn mở",
    "tech stack": "ngăn xếp công nghệ",
    "pull request": "pull request",
    "code review": "kiểm tra mã nguồn",
    "unit test": "kiểm thử đơn vị",
    "continuous integration": "tích hợp liên tục",
    "continuous deployment": "triển khai liên tục",
}

IT_GLOSSARY: Dict[str, str] = {
    "api": "API",
    "application": "ứng dụng",
    "app": "ứng dụng",
    "algorithm": "thuật toán",
    "ai": "AI",
    "authentication": "xác thực",
    "authorization": "phân quyền",
    "backend": "backend",
    "frontend": "frontend",
    "bug": "lỗi",
    "cache": "bộ nhớ đệm",
    "client": "máy khách",
    "cloud": "đám mây",
    "code": "mã",
    "compiler": "trình biên dịch",
    "container": "container",
    "cpu": "CPU",
    "database": "cơ sở dữ liệu",
    "dataset": "tập dữ liệu",
    "data": "dữ liệu",
    "debug": "gỡ lỗi",
    "deploy": "triển khai",
    "deployment": "quá trình triển khai",
    "developer": "lập trình viên",
    "devops": "DevOps",
    "endpoint": "điểm cuối",
    "encryption": "mã hóa",
    "firewall": "tường lửa",
    "framework": "framework",
    "function": "hàm",
    "gpu": "GPU",
    "hardware": "phần cứng",
    "host": "máy chủ",
    "inference": "suy luận",
    "input": "đầu vào",
    "interface": "giao diện",
    "kernel": "hạt nhân",
    "library": "thư viện",
    "memory": "bộ nhớ",
    "model": "mô hình",
    "network": "mạng",
    "output": "đầu ra",
    "parameter": "tham số",
    "password": "mật khẩu",
    "pipeline": "chuỗi xử lý",
    "port": "cổng",
    "process": "tiến trình",
    "programming": "lập trình",
    "protocol": "giao thức",
    "query": "truy vấn",
    "repository": "kho mã nguồn",
    "request": "yêu cầu",
    "response": "phản hồi",
    "runtime": "môi trường chạy",
    "script": "tập lệnh",
    "server": "máy chủ",
    "software": "phần mềm",
    "sql": "SQL",
    "stack": "ngăn xếp",
    "storage": "lưu trữ",
    "thread": "luồng",
    "token": "mã thông báo",
    "train": "huấn luyện",
    "training": "huấn luyện",
    "user": "người dùng",
    "vm": "máy ảo",
    "web": "web",
    "traffic": "lưu lượng",
    "latency": "độ trễ",
    "throughput": "thông lượng",
    "bandwidth": "băng thông",
    "performance": "hiệu năng",
    "scalability": "khả năng mở rộng",
    "availability": "tính sẵn sàng",
    "reliability": "độ tin cậy",
    "microservice": "microservice",
    "microservices": "microservices",
    "middleware": "middleware",
    "gateway": "cổng kết nối",
    "proxy": "proxy",
    "cluster": "cụm máy chủ",
    "node": "nút",
    "instance": "phần thể hiện",
    "orchestration": "điều phối",
    "kubernetes": "Kubernetes",
    "docker": "Docker",
}

# Context hints for ambiguous terms
AMBIGUOUS_TERMS: Dict[str, Dict[str, List[str]]] = {
    "bug": {
        "lỗi": ["code", "software", "system", "application", "program", "fix", "debug"],
    },
    "train": {
        "huấn luyện": ["model", "dataset", "data", "neural", "accuracy", "machine learning"],
    },
    "model": {
        "mô hình": ["train", "dataset", "predict", "accuracy", "learning", "inference"],
    },
    "port": {
        "cổng": ["server", "network", "tcp", "udp", "connection", "listen"],
    },
    "thread": {
        "luồng": ["process", "cpu", "parallel", "execution", "runtime", "background"],
    },
    "deploy": {
        "triển khai": ["server", "production", "application", "cloud", "release"],
    },
    "token": {
        "mã thông báo": ["auth", "api", "jwt", "security", "login", "validate"],
    },
    "query": {
        "truy vấn": ["sql", "database", "table", "select", "insert", "update"],
    },
}

# Priority 3 – Post-translation fixes (lowest priority)
# Each entry fixes a REAL observed mistranslation from the model.
POST_FIXES: Dict[str, str] = {
    "con lỗi": "lỗi",          
    "một con lỗi": "một lỗi",  
    "sâu bọ": "lỗi",            
    "con bọ": "lỗi",            
    "bọ": "lỗi",                
    "côn trùng": "lỗi",       
    "vi khuẩn": "lỗi",          
    "đào tạo mô hình": "huấn luyện mô hình",   
    "đào tạo người mẫu": "huấn luyện mô hình", 
    "máy phục vụ": "máy chủ",   
    "phục vụ": "máy chủ",      
    "cảng": "cổng",             
    "sợi chỉ": "luồng",         
    "mã nguồn nguồn": "mã nguồn",
    "cơ sở dữ liệu dữ liệu": "cơ sở dữ liệu",
    "triển khai ứng dụng tới máy chủ": "triển khai ứng dụng lên máy chủ",
    "triển khai ứng dụng đến máy chủ": "triển khai ứng dụng lên máy chủ",
    "giao thông cao": "lưu lượng cao",
    "giao thông": "lưu lượng",      
    "giao thông mạng": "lưu lượng mạng",
}

# Keywords that signal IT domain context
_IT_DOMAIN_KEYWORDS = {
    "code", "software", "program", "developer", "debug",
    "server", "database", "api", "deploy", "cloud",
    "network", "traffic", "microservice", "authentication",
    "performance", "latency", "throughput", "cluster",
}

# Flexible regex patterns for phrases that can be hyphenated or spaced.
# Maps canonical phrase key → regex that matches all surface variants.
FLEXIBLE_PHRASE_PATTERNS: Dict[str, str] = {
    # --- Core IT phrases (most important) ---
    "machine learning": r"\bmachine[\s-]+learning\b",
    "deep learning": r"\bdeep[\s-]+learning\b",
    "neural network": r"\bneural[\s-]+network\b",
    "artificial intelligence": r"\bartificial[\s-]+intelligence\b",
    "source code": r"\bsource[\s-]+code\b",
    "user interface": r"\buser[\s-]+interface\b",
    "virtual machine": r"\bvirtual[\s-]+machine\b",
    "operating system": r"\boperating[\s-]+system\b",
    "database query": r"\bdatabase[\s-]+query\b",
    "training data": r"\btraining[\s-]+data\b",
    "training dataset": r"\btraining[\s-]+dataset\b",
    "web server": r"\bweb[\s-]+server\b",
    "application server": r"\bapplication[\s-]+server\b",
    "cloud computing infrastructure": r"\bcloud[\s-]+computing[\s-]+infrastructure\b",
    "cloud computing": r"\bcloud[\s-]+computing\b",
    "computer network": r"\bcomputer[\s-]+network\b",
    "data structure": r"\bdata[\s-]+structure\b",
    "machine code": r"\bmachine[\s-]+code\b",
    "programming language": r"\bprogramming[\s-]+language\b",
    "object oriented programming": r"\bobject[\s-]+oriented[\s-]+programming\b",
    "sql query": r"\bsql[\s-]+query\b",
    "primary key": r"\bprimary[\s-]+key\b",
    "foreign key": r"\bforeign[\s-]+key\b",
    "runtime environment": r"\bruntime[\s-]+environment\b",
    "background process": r"\bbackground[\s-]+process\b",
    "application programming interface": r"\bapplication[\s-]+programming[\s-]+interface\b",
    # --- Networking & infrastructure ---
    "high traffic": r"\bhigh[\s-]+traffic\b",
    "network traffic": r"\bnetwork[\s-]+traffic\b",
    "load balancer": r"\bload[\s-]+balancer\b",
    "load balancing": r"\bload[\s-]+balancing\b",
    "system performance": r"\bsystem[\s-]+performance\b",
    "access control": r"\baccess[\s-]+control\b",
    "version control": r"\bversion[\s-]+control\b",
    # --- Architecture ---
    "open source": r"\bopen[\s-]+source\b",
    "tech stack": r"\btech[\s-]+stack\b",
    "pull request": r"\bpull[\s-]+request\b",
    "code review": r"\bcode[\s-]+review\b",
    "unit test": r"\bunit[\s-]+test\b",
    "continuous integration": r"\bcontinuous[\s-]+integration\b",
    "continuous deployment": r"\bcontinuous[\s-]+deployment\b",
}

# Patterns to detect wrong model output for phrases
PHRASE_OUTPUT_PATTERNS: Dict[str, List[str]] = {
    "source code": [r"mã(?:\s+nguồn|\s+code)?", r"code"],
    "machine learning": [r"học\s*máy|máy\s*học"],
    "deep learning": [r"học\s*sâu"],
    "neural network": [r"mạng\s*thần\s*kinh|mạng\s*nơ\s*-?ron"],
    "user interface": [r"giao\s*diện(?:\s*người\s*dùng|\s*người\s*sử\s*dụng)?"],
    "virtual machine": [r"máy\s*ảo"],
    "web server": [r"máy\s*chủ\s*web"],
    "application server": [r"máy\s*chủ\s*ứng\s*dụng"],
    "database query": [r"truy\s*vấn\s*(?:cơ\s*sở\s*dữ\s*liệu|database)", r"truy\s*vấn"],
    "sql query": [r"truy\s*vấn\s*sql|sql\s*query"],
    "training data": [r"dữ\s*liệu\s*huấn\s*luyện|tập\s*dữ\s*liệu\s*huấn\s*luyện"],
    "training dataset": [r"tập\s*dữ\s*liệu\s*huấn\s*luyện"],
    "cloud computing infrastructure": [r"cơ\s*sở\s*hạ\s*tầng\s*(?:điện\s*toán\s*)?đám\s*mây"],
    "cloud computing": [r"tính\s*toán\s*đám\s*mây", r"đám\s*mây", r"cloud[\s-]*computing"],
}

# Patterns to detect wrong model output for single terms.
# Each pattern captures a REAL observed wrong Vietnamese translation.
# Used by domain_overlay to fix model errors.
TERM_OUTPUT_PATTERNS: Dict[str, List[str]] = {
    "bug": [r"sâu\s*bọ|con\s*bọ|con\s*lỗi|bọ|côn\s*trùng|vi\s*khuẩn"],
    "train": [r"đào\s*tạo"],
    "model": [r"người\s*mẫu"],
    "server": [r"máy\s*phục\s*vụ"],
    "port": [r"cảng"],
    "thread": [r"sợi\s*chỉ"],
    "query": [r"hỏi\s*đáp"],
    "deploy": [r"đưa\s*lên|phát\s*hành"],
    "token": [r"thẻ"],
    "code": [r"mã\s*lệnh|mã\s*code"],
    "traffic": [r"giao\s*thông"],
}

NORMALIZATION_MAP: Dict[str, str] = {
    "mã code": "mã nguồn",
    "đoạn code": "mã nguồn",
    "bộ dữ liệu": "tập dữ liệu",
    "máy phục vụ web": "máy chủ web",
    "giao diện người sử dụng": "giao diện người dùng",
}

ACRONYM_MAP: Dict[str, str] = {
    "api": "API",
    "sql": "SQL",
    "cpu": "CPU",
    "gpu": "GPU",
    "ai": "AI",
    "devops": "DevOps",
}

# High-risk single terms that MUST be masked before sending to the model.
# These are terms the MarianMT model consistently mistranslates in IT context
# (e.g., "bug" → "con bọ", "input" → "đầu ra", "port" → "cảng").
# By masking them, the model translates the sentence structure without
# touching these terms, then we insert the correct Vietnamese after.
MASK_SINGLE_TERMS: Dict[str, str] = {
    # Ambiguous terms (model picks wrong meaning)
    "bug": "lỗi",
    "port": "cổng",
    "thread": "luồng",
    "token": "mã thông báo",
    # Terms model often confuses or swaps
    "input": "đầu vào",
    "output": "đầu ra",
    # Terms model translates with wrong sense
    "train": "huấn luyện",
    "training": "huấn luyện",
    "model": "mô hình",
    "deploy": "triển khai",
    "deployment": "quá trình triển khai",
    # Rule #6: traffic must be "lưu lượng", never "giao thông"
    "traffic": "lưu lượng",
}

# Common English suffixes for variant matching
_SUFFIX_STRIP = [
    ("ies", "y"),    # queries → query
    ("ses", "s"),    # processes → process (keep 's' for words ending in 'ss')
    ("es", "e"),     # interfaces → interface
    ("es", ""),      # fixes → fix
    ("s", ""),       # models → model
    ("ing", ""),     # training → train
    ("ing", "e"),    # deploying → deploy(e) — not needed but safe
    ("ed", ""),      # trained → train
    ("ed", "e"),     # deployed → deploy(e)
    ("tion", "te"),  # execution → execute
]


# ============================================================
# MASK MANAGER  (Problem #1 — was missing entirely)
# ============================================================

class MaskManager:
    """Protects special content from being modified by the model or postprocessing.

    Supports three layers:
      - special masks: URLs, code blocks, numbers, special tokens
      - phrase masks: glossary phrases masked before model, restored after
      - term masks: high-risk single IT terms masked before model
    """

    def __init__(self):
        self._special_map: Dict[str, str] = {}
        self._phrase_map: Dict[str, str] = {}
        self._term_map: Dict[str, str] = {}
        self._special_counter = 0
        self._phrase_counter = 0
        self._term_counter = 0

    def reset(self):
        self._special_map.clear()
        self._phrase_map.clear()
        self._term_map.clear()
        self._special_counter = 0
        self._phrase_counter = 0
        self._term_counter = 0


    def _next_special_token(self) -> str:
        self._special_counter += 1
        return f"XSPECIAL{self._special_counter}X"

    def mask_special(self, text: str) -> str:
        """Mask URLs, inline code, code blocks, and standalone numbers."""
        def _replace(match, store=self._special_map):
            token = self._next_special_token()
            store[token] = match.group(0)
            return token

        text = re.sub(r"```[\s\S]*?```", _replace, text)
        # Inline code (`...`)
        text = re.sub(r"`[^`]+`", _replace, text)
        # URLs
        text = re.sub(r"https?://\S+", _replace, text)
        # Standalone numbers (protect from model mangling)
        text = re.sub(r"\b\d+(?:\.\d+)?\b", _replace, text)
        return text

    def unmask_special(self, text: str) -> str:
        for token, original in self._special_map.items():
            text = text.replace(token, original)
        return text

    # --- Phrase masking (glossary phrases) ---

    def _next_phrase_token(self) -> str:
        self._phrase_counter += 1
        return f"XPHRASE{self._phrase_counter}X"

    def mask_phrases(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Mask glossary phrases in *source* text before sending to model.

        Returns (masked_text, phrase_token_to_vietnamese_map).
        Longer phrases are masked first to avoid partial matches.
        Uses flexible regex for known hyphenated variants (e.g. cloud-computing).
        """
        # Sort by length descending to mask longer phrases first
        for phrase in sorted(PHRASE_GLOSSARY.keys(), key=len, reverse=True):
            # Use flexible pattern if one exists (handles hyphens, etc.)
            if phrase in FLEXIBLE_PHRASE_PATTERNS:
                pattern = re.compile(FLEXIBLE_PHRASE_PATTERNS[phrase], re.IGNORECASE)
            else:
                pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
            while pattern.search(text):
                token = self._next_phrase_token()
                self._phrase_map[token] = PHRASE_GLOSSARY[phrase]
                # Replace only the first occurrence each iteration
                text = pattern.sub(token, text, count=1)
        return text, self._phrase_map

    def unmask_phrases(self, text: str, phrase_map: dict) -> str:
        for token, vi in phrase_map.items():
            text = text.replace(token, vi)
        return text

    # --- Single-term masking (high-risk IT terms) ---

    def _next_term_token(self) -> str:
        self._term_counter += 1
        return f"XTERM{self._term_counter}X"

    def mask_terms(self, text: str) -> str:
        """Mask high-risk single IT terms that the model commonly mistranslates.

        This prevents the model from translating words like 'bug', 'input',
        'output', 'port' incorrectly. The correct Vietnamese is inserted
        during unmask_terms().

        Must be called AFTER mask_phrases() to avoid masking terms that are
        already part of a masked phrase.
        """
        for term, vi in sorted(MASK_SINGLE_TERMS.items(),
                               key=lambda x: len(x[0]), reverse=True):
            pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            while pattern.search(text):
                token = self._next_term_token()
                self._term_map[token] = vi
                text = pattern.sub(token, text, count=1)
        return text

    def unmask_terms(self, text: str, term_map: dict) -> str:
        for token, vi in term_map.items():
            text = text.replace(token, vi)
        return text

# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_text(text: str) -> str:
    """Normalize whitespace and ensure space after sentence-ending punctuation."""
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    # CRITICAL: ensure every . ! ? is followed by a space before a letter
    text = re.sub(r"([.!?])([A-Za-z\u00C0-\u024F\u1E00-\u1EFF])", r"\1 \2", text)
    return text


def _chunk_unpunctuated_text(text: str, min_words: int = 8,
                              max_words: int = 12) -> str:
    """If text has no sentence-ending punctuation, split into chunks of
    8–12 words joined by '. ' to prevent long unbroken model input."""
    if re.search(r"[.!?]", text):
        return text  # already has punctuation — leave it
    words = text.split()
    if len(words) <= max_words:
        return text  # short enough — no chunking needed
    chunks: List[str] = []
    i = 0
    while i < len(words):
        end = min(i + max_words, len(words))
        # Try to find a natural break between min and max
        if end < len(words) and (end - i) > min_words:
            chunk = " ".join(words[i:end])
        else:
            chunk = " ".join(words[i:end])
        chunks.append(chunk)
        i = end
    return ". ".join(chunks) + "."


def split_sentences(text: str) -> List[str]:
    """Split text into sentences at TRUE sentence boundaries only.

    Guards against false splits caused by:
      - Decimal numbers: "version 3.5 is stable"
      - Abbreviations:   "e.g. this works" / "i.e. something"
      - Single-letter:   "U.S. government"

    Only splits when:
      - .!? is preceded by 2+ letter-word (not a single letter / digit)
      - .!? is followed by a space + uppercase letter, OR is at end of string

    For very long comma-separated clauses (>15 words without punctuation),
    splits at commas to help the model produce better translations.
    """
    text = text.strip()
    if not text:
        return []

    # Step 1: Split at true sentence boundaries
    # Pattern: a word of 2+ chars, then .!?, then space + uppercase letter
    # This avoids splitting on "3.5", "e.g.", "U.S.", etc.
    sentence_boundary = re.compile(
        r"(?<=[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF]{2}[.!?])"  # 2+ letter word + punct
        r"\s+"                                                  # whitespace
        r"(?=[A-Z\u00C0-\u024F\u1E00-\u1EFF])"                # next sentence starts uppercase
    )
    parts = sentence_boundary.split(text)
    parts = [p.strip() for p in parts if p.strip()]

    # Step 2: For any part that is very long with no sentence-ending punctuation,
    # split at commas to produce manageable chunks for the model.
    MAX_WORDS_BEFORE_COMMA_SPLIT = 15
    final_parts: List[str] = []
    for part in parts:
        words = part.split()
        if len(words) > MAX_WORDS_BEFORE_COMMA_SPLIT and not re.search(r"[.!?]", part[:-1]):
            # Split at commas, but only keep chunks that are meaningful (3+ words)
            clauses = re.split(r",\s*", part)
            buffer: List[str] = []
            for clause in clauses:
                buffer.append(clause.strip())
                joined = ", ".join(buffer)
                if len(joined.split()) >= 6:
                    final_parts.append(joined)
                    buffer = []
            if buffer:
                remainder = ", ".join(buffer)
                if final_parts:
                    # Merge short trailing clause with previous part
                    final_parts[-1] = final_parts[-1] + ", " + remainder
                else:
                    final_parts.append(remainder)
        else:
            final_parts.append(part)

    return [p for p in final_parts if p.strip()]


def _stem_to_base(word: str) -> Optional[str]:
    """Try to reduce an English word to its base form using suffix stripping.

    Returns the base form if it matches a glossary entry, else None.
    """
    word_lower = word.lower()
    # Direct match — no stemming needed
    if word_lower in IT_GLOSSARY or word_lower in PHRASE_GLOSSARY:
        return word_lower

    for suffix, replacement in _SUFFIX_STRIP:
        if word_lower.endswith(suffix):
            candidate = word_lower[: -len(suffix)] + replacement
            if candidate in IT_GLOSSARY:
                return candidate
    return None


# ============================================================
# GLOSSARY DETECTION  (with variant support — Problem #8)
# ============================================================

def find_glossary_terms(text: str) -> List[str]:
    """Find glossary terms in text, including morphological variants.

    Case-insensitive: works for 'cloud computing', 'CLOUD COMPUTING', etc.
    Uses FLEXIBLE_PHRASE_PATTERNS for hyphenated/variant phrases.
    """
    lower_text = text.lower()
    found: List[str] = []

    # Check phrase glossary first (higher priority, longest first)
    for term in sorted(PHRASE_GLOSSARY.keys(), key=len, reverse=True):
        # Use flexible pattern if available (handles hyphens, etc.)
        if term in FLEXIBLE_PHRASE_PATTERNS:
            pattern = FLEXIBLE_PHRASE_PATTERNS[term]
        else:
            pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, lower_text, flags=re.IGNORECASE):
            found.append(term)

    # Check single-term glossary (with variant support)
    for term in sorted(IT_GLOSSARY.keys(), key=len, reverse=True):
        if term in found:
            continue
        # Direct match (case-insensitive)
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, lower_text, flags=re.IGNORECASE):
            found.append(term)
            continue
        # Variant match: check each word in text
        for word in re.findall(r"[a-zA-Z]+", text):
            base = _stem_to_base(word)
            if base == term and term not in found:
                found.append(term)
                break

    return found


# ============================================================
# CONTEXT-AWARE SCORING  (Problem #4 — use word boundaries)
# ============================================================

def score_term_meaning(term: str, sentence_lower: str) -> str:
    """Choose the best Vietnamese translation for an ambiguous term
    based on co-occurring context words (using word-boundary regex).

    Safeguard: always lowercases sentence_lower to ensure case-insensitive
    matching even if caller passes mixed-case text.
    """
    sentence_lower = sentence_lower.lower()  # safeguard
    if term in AMBIGUOUS_TERMS:
        candidate_scores: Dict[str, int] = {}
        for meaning, hints in AMBIGUOUS_TERMS[term].items():
            score = 0
            for hint in hints:
                # Use word boundaries to avoid partial matches
                if re.search(r"\b" + re.escape(hint) + r"\b", sentence_lower):
                    score += 1
            candidate_scores[meaning] = score
        best = max(candidate_scores, key=candidate_scores.get)
        if candidate_scores[best] > 0:
            return best
    return IT_GLOSSARY.get(term, term)


# ============================================================
# SHORT TEXT HANDLING  (Problem #2 — only single-word bypass)
# ============================================================

def is_single_word(text: str) -> bool:
    """Only TRUE single-word input bypasses the model."""
    words = text.strip().split()
    return len(words) == 1


def translate_single_word(word: str) -> str:
    """Translate a single word directly from glossary (no model)."""
    word_lower = word.lower().strip()
    # Try direct glossary lookup
    if word_lower in IT_GLOSSARY:
        return IT_GLOSSARY[word_lower]
    # Try variant stemming
    base = _stem_to_base(word_lower)
    if base and base in IT_GLOSSARY:
        return IT_GLOSSARY[base]
    # Check phrase glossary (unlikely for single word, but safe)
    if word_lower in PHRASE_GLOSSARY:
        return PHRASE_GLOSSARY[word_lower]
    return word


def translate_word_by_word(text: str) -> str:
    """Word-by-word dictionary translation (for 'Dịch từ / ngắn' mode)."""
    words = text.strip().split()
    result = []
    for word in words:
        clean = re.sub(r"[^a-zA-Z]", "", word).lower()
        if clean:
            translated = translate_single_word(clean)
            trailing = re.findall(r"[^a-zA-Z]+$", word)
            result.append(translated + (trailing[0] if trailing else ""))
        else:
            result.append(word)
    return " ".join(result)


# ============================================================
# MODEL INTERFACE
# ============================================================

def tokenize_for_model(text: str, tokenizer: MarianTokenizer):
    return tokenizer([text], return_tensors="pt", padding=True,
                     truncation=True, max_length=512)


def model_generate(inputs, model):
    return model.generate(**inputs, max_length=512, num_beams=5,
                          early_stopping=True)


def decode_model_output(generated, tokenizer: MarianTokenizer) -> str:
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


def translate_with_model(text: str, tokenizer: MarianTokenizer,
                         model: MarianMTModel) -> str:
    inputs = tokenize_for_model(text, tokenizer)
    generated = model_generate(inputs, model)
    return decode_model_output(generated, tokenizer)


# ============================================================
# MODEL CORRUPTION DETECTION & FALLBACK
# ============================================================

def _is_corrupted_output(translated: str) -> bool:
    """Detect model hallucination / corruption.

    Returns True if:
      - output contains the word 'name' (common hallucination)
      - any single token is repeated more than 5 times
    """
    if re.search(r"\bname\b", translated, re.IGNORECASE):
        return True
    tokens = translated.split()
    if tokens:
        from collections import Counter
        counts = Counter(tokens)
        if counts.most_common(1)[0][1] > 5:
            return True
    return False


def _denoise_mangled_tokens(text: str, mask_mgr: 'MaskManager') -> str:
    """Fix model-mangled mask tokens (e.g., XPHRIRS2X → XPHRASE2X).
    
    The model sometimes tries to "translate" mask tokens, mangling them.
    This function uses fuzzy matching to recover original token names.
    """
    # Find all mangled tokens in text (pattern: X followed by letters/numbers, ending with X)
    mangled_pattern = re.compile(r"\bX[A-Z]+\d+X\b", re.IGNORECASE)
    
    for mangled in mangled_pattern.findall(text):
        # Try to find the closest original token in the mask maps
        best_match = None
        best_ratio = 0
        
        # Check against phrase tokens
        for original_token in mask_mgr._phrase_map.keys():
            # Simple similarity: count matching characters
            ratio = sum(1 for a, b in zip(original_token, mangled) if a == b) / max(len(original_token), len(mangled))
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = original_token
        
        # Check against term tokens
        for original_token in mask_mgr._term_map.keys():
            ratio = sum(1 for a, b in zip(original_token, mangled) if a == b) / max(len(original_token), len(mangled))
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = original_token
        
        # If we found a likely match (>50% similar), replace it
        if best_match and best_ratio > 0.5:
            text = text.replace(mangled, best_match)
    
    return text


def _dictionary_fallback(sentence: str) -> str:
    """Translate a sentence word-by-word / phrase-by-phrase from glossaries.

    Used as a fallback when the model produces corrupted output.
    Tries phrase glossary first (longest match), then single words.
    """
    result = sentence
    # Replace phrases first (longest first)
    for phrase, vi in sorted(PHRASE_GLOSSARY.items(),
                             key=lambda x: len(x[0]), reverse=True):
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        result = pattern.sub(vi, result)
    # Replace remaining English words from IT glossary
    def _replace_word(match):
        word = match.group(0)
        lower = word.lower()
        if lower in IT_GLOSSARY:
            return IT_GLOSSARY[lower]
        base = _stem_to_base(lower)
        if base and base in IT_GLOSSARY:
            return IT_GLOSSARY[base]
        return word
    result = re.sub(r"\b[a-zA-Z]+\b", _replace_word, result)
    return result


# ============================================================
# DOMAIN OVERLAY  (Problem #5 — only fix when wrong)
# ============================================================

def domain_overlay(source_sentence: str, translated_sentence: str) -> str:
    """Fix incorrect translations in model output.

    Rules:
      - Only act when the English term still appears in output OR
        a known-wrong Vietnamese pattern is detected.
      - NEVER overwrite already-correct Vietnamese phrases.
    """
    source_lower = source_sentence.lower()
    output = translated_sentence

    # --- Pass 1: Fix phrases (highest priority, longest first) ---
    for phrase, vi in sorted(PHRASE_GLOSSARY.items(),
                             key=lambda x: len(x[0]), reverse=True):
        # Use flexible pattern for source matching if available
        if phrase in FLEXIBLE_PHRASE_PATTERNS:
            src_pat = FLEXIBLE_PHRASE_PATTERNS[phrase]
        else:
            src_pat = r"\b" + re.escape(phrase) + r"\b"
        if not re.search(src_pat, source_lower):
            continue
        # If correct translation already present, skip
        if vi in output:
            continue
        # If English phrase leaked through (including hyphenated), replace it
        if phrase in FLEXIBLE_PHRASE_PATTERNS:
            en_pattern = re.compile(FLEXIBLE_PHRASE_PATTERNS[phrase], re.IGNORECASE)
        else:
            en_pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        if en_pattern.search(output):
            output = en_pattern.sub(vi, output)
            continue
        # If a known-wrong pattern is detected, fix it
        patterns = PHRASE_OUTPUT_PATTERNS.get(phrase, [])
        for pat in patterns:
            if re.search(pat, output, flags=re.IGNORECASE):
                output = re.sub(pat, vi, output, count=1, flags=re.IGNORECASE)
                break

    # --- Pass 2: Fix single terms (only if wrong) ---
    for term in sorted(IT_GLOSSARY.keys(), key=len, reverse=True):
        if not re.search(r"\b" + re.escape(term) + r"\b", source_lower):
            continue
        chosen_vi = score_term_meaning(term, source_lower)
        # If correct Vietnamese already present, skip entirely
        if chosen_vi in output:
            continue
        # If English term leaked through, replace
        en_pat = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        if en_pat.search(output):
            output = en_pat.sub(chosen_vi, output)
            continue
        # If a known-wrong pattern exists, fix it
        wrong_patterns = TERM_OUTPUT_PATTERNS.get(term, [])
        for pat in wrong_patterns:
            if re.search(pat, output, flags=re.IGNORECASE):
                output = re.sub(pat, chosen_vi, output, count=1,
                                flags=re.IGNORECASE)
                break

    # --- Pass 3: Light post-fixes ---
    SIMPLE_FIXES = {
        r"sửa lỗi trong mã\b": "sửa lỗi trong mã nguồn",
        r"triển khai ứng dụng (?:tới|đến) máy chủ":
            "triển khai ứng dụng lên máy chủ",
        r"tối ưu truy vấn SQL cho hiệu suất tốt hơn":
            "tối ưu truy vấn SQL để cải thiện hiệu năng",
    }
    for pat, repl in SIMPLE_FIXES.items():
        output = re.sub(pat, repl, output, flags=re.IGNORECASE)

    # --- Pass 4: IT domain enforcement ---
    source_words = set(re.findall(r"[a-zA-Z]+", source_lower))
    if source_words & _IT_DOMAIN_KEYWORDS:
        # In IT context: "code" → "mã nguồn", "bug" → "lỗi"
        output = re.sub(r"\bmã\b(?!\s*nguồn)", "mã nguồn", output)
        # "mật mã" → "mã nguồn" ONLY if source contains "code"
        if "code" in source_words:
            output = re.sub(r"mật\s*mã", "mã nguồn", output, flags=re.IGNORECASE)

    return output


# ============================================================
# POSTPROCESSING
# ============================================================

def local_postprocess_vi(text: str) -> str:
    """Per-sentence cleanup after domain overlay."""
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    for src, dst in NORMALIZATION_MAP.items():
        text = re.sub(re.escape(src), dst, text, flags=re.IGNORECASE)

    for src, dst in ACRONYM_MAP.items():
        text = re.sub(r"\b" + re.escape(src) + r"\b", dst, text,
                      flags=re.IGNORECASE)
    return text


def postprocess_vi(text: str, source_text: str = "") -> str:
    """Global postprocessing on the final merged translation.

    Args:
        text: The translated Vietnamese text.
        source_text: Original English text (used for context-aware fixes).
    """
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Apply post-fixes (longest first to avoid partial matches)
    for src, dst in sorted(POST_FIXES.items(), key=lambda x: len(x[0]),
                           reverse=True):
        text = re.sub(re.escape(src), dst, text, flags=re.IGNORECASE)

    for src, dst in NORMALIZATION_MAP.items():
        text = re.sub(re.escape(src), dst, text, flags=re.IGNORECASE)

    for src, dst in ACRONYM_MAP.items():
        text = re.sub(r"\b" + re.escape(src) + r"\b", dst, text,
                      flags=re.IGNORECASE)

    # Context-aware fix: "mật mã" → "mã nguồn" only when source has "code"
    if source_text and re.search(r"\bcode\b", source_text, re.IGNORECASE):
        text = re.sub(r"mật\s*mã", "mã nguồn", text, flags=re.IGNORECASE)

    # Ensure space after punctuation (never merge sentences)
    text = re.sub(r"([.!?])([A-Za-z\u00C0-\u024F\u1E00-\u1EFF])", r"\1 \2", text)

    if text:
        text = text[0].upper() + text[1:]
    return text


# ============================================================
# MAIN PIPELINE  (Problem #6 — correct order, no conflicts)
# ============================================================

def translate_it_text(text: str, tokenizer: MarianTokenizer,
                      model: MarianMTModel) -> Tuple[str, List[str]]:
    """Main translation pipeline for IT text.

    Steps:
      1.  Mask special content (URLs, code, numbers)
      2.  Normalize text (incl. space after punctuation)
      3.  Chunk unpunctuated text into manageable pieces
      4.  Split into sentences
      Per sentence:
        5.  Mask glossary phrases → placeholder tokens
        6.  Mask high-risk single terms → placeholder tokens
        7.  Run model on masked input
        7b. Denoise: Fix model-mangled mask tokens (e.g. XPHRIRS2X → XPHRASE2X)
        7c. Corruption guard → fallback to dictionary if bad output
        8a. Unmask terms → insert correct Vietnamese
        8b. Unmask phrases → insert correct Vietnamese
        9.  Domain overlay (fix remaining errors only)
        10. Local postprocess
      11. Merge and global postprocess (with source context)
      12. Unmask special content
    """
    # Create a fresh mask manager for this translation
    mask_mgr = MaskManager()

    # [1] MASK SPECIAL CONTENT
    masked_text = mask_mgr.mask_special(text)

    # [2] NORMALIZE (ensures space after .!? before letters)
    clean_text = normalize_text(masked_text)
    if not clean_text:
        return "", []

    # [3] CHUNK unpunctuated text to prevent long unbroken model input
    clean_text = _chunk_unpunctuated_text(clean_text)

    # [4] DETECT GLOSSARY TERMS (for reporting to UI)
    found_terms = find_glossary_terms(clean_text)

    # [5] SPLIT SENTENCES (uses \s* to handle missing spaces)
    sentences = split_sentences(clean_text)
    translated_sentences: List[str] = []

    for sentence in sentences:
        # Create a fresh mask manager for each sentence
        sent_mask_mgr = MaskManager()

        # [5] MASK PHRASES (return map riêng cho từng sentence)
        phrase_masked, phrase_map = sent_mask_mgr.mask_phrases(sentence)

        # [6] MASK TERMS (dùng text đã mask phrase)
        term_masked = sent_mask_mgr.mask_terms(phrase_masked)

        # [7] RUN MODEL on masked text
        raw_translated = translate_with_model(term_masked, tokenizer, model)

        # [7b] DENOISE: Fix model-mangled mask tokens
        raw_translated = _denoise_mangled_tokens(raw_translated, sent_mask_mgr)

        # [7c] CORRUPTION GUARD — if model output is bad, use dictionary
        if _is_corrupted_output(raw_translated):
            raw_translated = _dictionary_fallback(sentence)

        # [8a] UNMASK terms → insert correct Vietnamese for single terms
        unmasked = sent_mask_mgr.unmask_terms(raw_translated, sent_mask_mgr._term_map)

        # [8b] UNMASK phrases → insert correct Vietnamese for phrases
        unmasked = sent_mask_mgr.unmask_phrases(unmasked, phrase_map)

        # [9] DOMAIN OVERLAY (only fix incorrect)
        overlayed = domain_overlay(sentence, unmasked)

        # [10] LOCAL POSTPROCESS
        local_result = local_postprocess_vi(overlayed)

        translated_sentences.append(local_result)

    # [11] MERGE
    final_text = " ".join(translated_sentences)

    # [12] GLOBAL POSTPROCESS (pass source for context-aware fixes)
    final_text = postprocess_vi(final_text, source_text=text)

    # [13] UNMASK SPECIAL CONTENT
    final_text = mask_mgr.unmask_special(final_text)

    return final_text, found_terms