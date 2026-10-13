"""
================================================================================
HỆ THỐNG GỢI Ý PHỐI ĐỒ THỜI TRANG THÔNG MINH (AI SIC CAPSTONE)
CỔNG KHỞI ĐỘNG ỨNG DỤNG THỜI TRANG (UNIVERSAL APPLICATION ENTRYPOINT)
Tệp: app_web/app.py
================================================================================
Hỗ trợ linh hoạt 2 cách khởi chạy:
1. Chạy Web HTML/CSS/JS (Khuyên dùng):
   python app_web/app.py   (hoặc python app_web/run_web.py)
   -> Tự động khởi động máy chủ Flask và mở trình duyệt tại http://127.0.0.1:5000

2. Chạy giao diện Streamlit (Nếu muốn):
   streamlit run app_web/app.py   (hoặc streamlit run app_web/streamlit_app.py)
   -> Khởi động giao diện Streamlit tại http://localhost:8501
================================================================================
"""

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Kiểm tra môi trường thực thi
def is_running_under_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_context
        return get_script_run_context() is not None
    except Exception:
        return False

if is_running_under_streamlit():
    # Được chạy bởi 'streamlit run app_web/app.py'
    import streamlit_app
else:
    # Được chạy bởi 'python app_web/app.py'
    import run_web
