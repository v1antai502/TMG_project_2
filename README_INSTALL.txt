TMG301 - Clickbait Intelligence: Giao diện website cho macOS

YÊU CẦU:
- Đang ở repo TMG_project_re-main có service/app.py
- Python .venv đã được cài và checkpoint outputs/phobert_final/best.pt đã tồn tại

CÀI ĐẶT:
1. Giải nén ZIP, rồi chép thư mục web và file install_web.py vào TMG_project_re-main.
2. Mở Terminal ở thư mục gốc TMG_project_re-main.
3. source .venv/bin/activate
4. python install_web.py
5. Nếu server cũ đang chạy, nhấn Control+C tại Terminal server.
6. MODEL_DIR=outputs/phobert_final python -m service.app
7. Truy cập http://127.0.0.1:8000/

Trang web gửi POST /predict cùng origin với FastAPI: không cần Node.js, npm, CORS hay cài thư viện mới.
Có form nhập tiêu đề và lead, ví dụ, kết quả trực quan, phần batch tối đa 30 tiêu đề.
Nếu muốn khôi phục service cũ, chép nội dung service/app.py.before_web về service/app.py.
