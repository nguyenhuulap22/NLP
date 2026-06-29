# IT Translator Pro 💻

Dự án dịch máy tự động chuyên ngành Công nghệ thông tin (IT) từ tiếng Anh sang tiếng Việt. Hệ thống sử dụng mô hình học sâu kết hợp với kỹ thuật **Constrained Decoding** (Ép buộc sinh từ vựng) để đảm bảo các thuật ngữ chuyên ngành được dịch chính xác 100% dựa trên từ điển cho trước.

## 🌟 Điểm nổi bật
- **Mô hình cốt lõi:** Sử dụng kiến trúc Transformer (mô hình `Helsinki-NLP/opus-mt-en-vi` của Hugging Face).
- **Constrained Beam Search:** Can thiệp sâu vào quá trình sinh từ (decoding) bằng thuật toán FSA (Finite State Automaton) để bắt buộc mô hình sinh ra đúng thuật ngữ tiếng Việt thay vì dịch tự do.
- **Tiền xử lý thông minh:** Tích hợp `EntityNormalizer` giúp che giấu (mask) và bảo vệ nguyên vẹn các đường link (URL), địa chỉ email, số điện thoại hay đường dẫn file trong quá trình dịch.
- **Xử lý linh hoạt (Hybrid Fallback):** Tự động thử nghiệm dịch tự nhiên trước, sau đó gợi ý từ điển, và chỉ tung ra "vũ khí" ép từ vựng ở mức cao nhất khi mô hình cơ sở liên tục dịch sai.
- **Giao diện:** Tích hợp ứng dụng Web trực quan xây dựng bằng **Streamlit**.

## 🛠️ Cài đặt

1. **Clone repository về máy:**
   ```bash
   git clone https://github.com/nguyenhuulap22/NLP.git
   cd NLP
   ```

2. **Cài đặt thư viện (Dependencies):**
   Yêu cầu Python 3.8+ và các thư viện hỗ trợ AI như `torch`, `transformers`, `streamlit`.
   ```bash
   pip install torch transformers streamlit
   ```

## 🚀 Hướng dẫn sử dụng

Chạy ứng dụng web bằng Streamlit:
```bash
streamlit run app.py
```
Sau khi chạy lệnh, trình duyệt sẽ tự động mở lên giao diện để bạn nhập câu tiếng Anh và xem kết quả dịch thuật.

## 📂 Cấu trúc thư mục chính

- `app.py` & `ui.py`: Giao diện ứng dụng Streamlit.
- `models/`: Chứa code tải và cấu hình mô hình (tắt cache, bật output attention).
- `decoding/`: Chứa thuật toán Constrained Beam Search ép từ cốt lõi.
- `preprocessing/`: Xử lý văn bản, dọn dẹp unicode, che các thực thể đặc biệt (URL, Email).
- `terminology/`: Quản lý từ điển thuật ngữ (Glossary) và dò tìm từ vựng trong câu đầu vào.
- `constraints/`: Xây dựng trạng thái FSA (Finite State Automaton) dựa trên từ khóa.

## 🤝 Tác giả
- Nguyễn Hữu Lập
