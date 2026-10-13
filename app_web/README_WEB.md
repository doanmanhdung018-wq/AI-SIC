# 🌐 THƯ MỤC APP_WEB - HỆ THỐNG GỢI Ý PHỐI ĐỒ THỜI TRANG

> **Ghi chú bảo vệ đồ án:** Thư mục này phục vụ mục đích chuyển hóa thuật toán lõi AI thành sản phẩm thực tế. Học viên báo cáo trước Hội đồng rằng phần giao diện Web và các mô-đun kết nối này được phát triển nhanh với sự hỗ trợ của các công cụ AI (ChatGPT, Antigravity) để tối ưu thời gian, trong khi học viên nắm chắc 100% thuật toán Deep Learning lõi trong thư mục `Model/`.

---

## 📁 1. Cấu trúc thư mục Giao diện Web (Rõ ràng & Tách bạch Frontend/Backend)

```
app_web/
│
├── templates/                      <-- [FRONTEND HTML]
│   └── index.html                  <-- Giao diện Web tương tác cao cấp (HTML5 chuẩn)
│
├── static/                         <-- [FRONTEND TÀI NGUYÊN TĨNH]
│   ├── css/
│   │   └── style.css               <-- Định kiểu giao diện Glassmorphism, Responsive
│   └── js/
│       └── main.js                 <-- JavaScript xử lý Drag & Drop, AJAX gọi API, render
│
├── backend.py                      <-- [BACKEND API] Máy chủ Flask REST API
├── model_caller.py                 <-- [INFERENCE] Nạp embeddings & tính tương đồng Cosine
├── run_web.py                      <-- [RUNNER] Script khởi chạy Web Server và tự mở trình duyệt
├── start_web.bat                   <-- [WINDOWS 1-CLICK] Nhấp đúp chuột để mở Web ngay lập tức
├── streamlit_app.py                <-- [STREAMLIT] Giao diện Streamlit tương tác thay thế
├── app.py                          <-- [ENTRYPOINT] Cầu nối thông minh (Hỗ trợ cả Web & Streamlit)
└── requirements.txt                <-- Thư viện phụ thuộc riêng cho Web
```

---

## 🚀 2. Các cách khởi chạy Ứng dụng Web

### Cách 1: Nhấp đúp chuột (Dễ nhất trên Windows)
- Vào thư mục `app_web/` và **nhấp đúp chuột vào tệp `start_web.bat`**.
- Hệ thống sẽ tự động khởi động server và mở trình duyệt tại địa chỉ `http://127.0.0.1:5000`.

### Cách 2: Chạy lệnh Python chuẩn
```bash
python app_web/run_web.py
```
*Trình duyệt sẽ tự động bật lên với đầy đủ tính năng chọn mẫu từ kho hoặc upload ảnh cá nhân.*

### Cách 3: Chạy phiên bản Streamlit (Nếu muốn)
```bash
streamlit run app_web/streamlit_app.py
```
*Mở giao diện phân tích dữ liệu trên cổng `http://localhost:8501`.*

---

## 💡 3. Các tính năng nổi bật trên Giao diện Web
1. **Kho 1,200 sản phẩm thật**: Chọn nhanh các mẫu trang phục đại diện để xem gợi ý tức thì mà không cần tìm ảnh.
2. **Tải ảnh cá nhân (Upload)**: Kéo thả hoặc chọn ảnh trang phục bất kỳ từ máy tính; mô hình ResNet50 sẽ trích xuất vector đặc trưng trong 25ms.
3. **Bộ lọc thông minh**: Chọn giới tính (Nam/Nữ), hoàn cảnh (Dạo phố/Công sở/Thể thao), và chuyển đổi giữa thuật toán ResNet50 (2048D) và Baseline HSV (512D) để chứng minh độ vượt trội của Deep Learning.
4. **Hiển thị trực quan**: Cột trái hiển thị món đồ gốc, cột phải hiển thị bộ outfit bù trừ hoàn chỉnh với thanh đo % tương thích phong cách.
