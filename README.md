# IT Translator Pro

## Giới thiệu

**IT Translator Pro** là công cụ hỗ trợ dịch thuật Anh - Việt dành cho chuyên ngành Công nghệ thông tin. Chương trình giúp người dùng dịch các từ, cụm từ hoặc câu tiếng Anh có chứa thuật ngữ kỹ thuật sang tiếng Việt dễ hiểu hơn.

Công cụ phù hợp với sinh viên ngành Công nghệ thông tin, người học lập trình hoặc người thường xuyên đọc tài liệu tiếng Anh chuyên ngành.

## Chức năng chính

- Dịch văn bản tiếng Anh sang tiếng Việt.
- Hỗ trợ dịch thuật ngữ chuyên ngành CNTT.
- Nhập văn bản trực tiếp.
- Nhập nội dung bằng giọng nói.
- Tải tệp văn bản để dịch.
- Hiển thị kết quả dịch tiếng Việt.
- Xóa nội dung đã nhập.
- Lưu lịch sử dịch.
- Cung cấp ví dụ nhanh để người dùng thử nghiệm.

## Một số thuật ngữ được hỗ trợ

| Thuật ngữ tiếng Anh | Nghĩa trong CNTT |
|---|---|
| bug | lỗi phần mềm |
| database | cơ sở dữ liệu |
| server | máy chủ |
| model | mô hình |
| dataset | tập dữ liệu |
| application | ứng dụng |
| framework | bộ khung phát triển |
| training | huấn luyện |

## Công nghệ sử dụng

- Python
- Streamlit
- MarianMT
- Transformers
- Torch
- SentencePiece
- Từ điển thuật ngữ chuyên ngành CNTT

## Cấu trúc thư mục

```text
NLP_T/
│
├── app.py              # File chạy chính của chương trình
├── ui.py               # Xử lý giao diện chương trình
├── dictionary.py       # Từ điển thuật ngữ CNTT
├── requirements.txt    # Danh sách thư viện cần cài đặt
├── .gitignore          # Các file/thư mục không đưa lên GitHub
└── README.md           # Mô tả dự án
Cài đặt chương trình
Bước 1: Tải project về máy
git clone https://github.com/nguyenhuulap22/NLP.git

Sau đó mở thư mục project:

cd NLP
Bước 2: Tạo môi trường ảo
python -m venv venv

Kích hoạt môi trường ảo trên Windows:

venv\Scripts\activate
Bước 3: Cài đặt thư viện
pip install -r requirements.txt
Chạy chương trình

Chạy lệnh sau trong terminal:

streamlit run app.py
