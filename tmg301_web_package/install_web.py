"""Add a local website to the existing TMG301 FastAPI service.
Run from the repository root: python install_web.py
Keeps existing /health and /predict routes untouched.
"""
from pathlib import Path

root = Path(__file__).resolve().parent
app_file = root / 'service' / 'app.py'
html = root / 'web' / 'index.html'
if not app_file.is_file():
    raise SystemExit('Không tìm thấy service/app.py. Hãy chép gói này vào thư mục gốc TMG_project_re-main.')
if not html.is_file():
    raise SystemExit('Không tìm thấy web/index.html. Hãy giải nén toàn bộ ZIP.')
source = app_file.read_text(encoding='utf-8')
marker = '# TMG301 WEB UI ROUTE (added by install_web.py)'
if marker in source:
    print('Giao diện web đã được cài trước đó. Không cần chỉnh sửa thêm.')
    raise SystemExit(0)
if 'app = FastAPI(' not in source:
    raise SystemExit('Không nhận diện được cấu trúc FastAPI. Không sửa file.')
backup = app_file.with_name('app.py.before_web')
if not backup.exists():
    backup.write_text(source, encoding='utf-8')
# Insert before the existing __main__ block. This only creates GET /.
addition = '''\n\n# TMG301 WEB UI ROUTE (added by install_web.py)\nfrom fastapi.responses import FileResponse\n\n@app.get("/", include_in_schema=False)\ndef web_home():\n    return FileResponse(Path(__file__).resolve().parent.parent / "web" / "index.html")\n\n'''
needle = '\nif __name__ == "__main__":'
if needle not in source:
    raise SystemExit('Không tìm thấy điểm chèn an toàn trong service/app.py. Không sửa file.')
source = source.replace(needle, addition + needle, 1)
app_file.write_text(source, encoding='utf-8')
print('✅ Đã tạo website tại http://127.0.0.1:8000/')
print('✅ Giữ nguyên /health, /predict, /docs và Chrome Extension.')
print(f'✅ Bản sao lưu: {backup}')
print('Khởi động lại server: MODEL_DIR=outputs/phobert_final python -m service.app')
