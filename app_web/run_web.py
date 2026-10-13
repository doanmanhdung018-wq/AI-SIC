"""
================================================================================
SCRIPT KHỞI CHẠY ỨNG DỤNG WEB THỜI TRANG (AI SIC CAPSTONE)
Chạy lệnh: python app_web/run_web.py
Hoặc nhấp đúp vào: app_web/start_web.bat
================================================================================
"""

import os
import sys
import webbrowser
import threading
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from backend import app

def open_browser():
    time.sleep(1.2)
    url = "http://127.0.0.1:5000"
    print(f"\n🌐 Đang tự động mở trình duyệt tại: {url}")
    webbrowser.open(url)

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 KHỞI ĐỘNG HỆ THỐNG GỢI Ý PHỐI ĐỒ THỜI TRANG THÔNG MINH (AI SIC)")
    print("📁 Frontend HTML/CSS/JS: app_web/templates/ và app_web/static/")
    print("📁 Backend API: app_web/backend.py và app_web/model_caller.py")
    print("🌍 Địa chỉ truy cập: http://127.0.0.1:5000")
    print("=" * 70)

    # Mở trình duyệt trên luồng riêng
    threading.Thread(target=open_browser, daemon=True).start()

    # Chạy máy chủ Flask
    app.run(host="127.0.0.1", port=5000, debug=False)
