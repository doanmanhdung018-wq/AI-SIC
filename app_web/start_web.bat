@echo off
chcp 65001 > nul
title AI Fashion Recommender - Samsung Innovation Campus
echo ======================================================================
echo   HỆ THỐNG GỢI Ý PHỐI ĐỒ THỜI TRANG THÔNG MINH (AI SIC CAPSTONE)
echo   Tác giả: Đoàn Mạnh Dũng
echo ======================================================================
echo.
echo [1/2] Đang khởi động Web Server và nạp mô hình AI...
echo [2/2] Trình duyệt web sẽ tự động mở tại: http://127.0.0.1:5000
echo.
python "%~dp0run_web.py"
pause
