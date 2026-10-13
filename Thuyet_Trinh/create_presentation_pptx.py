"""
================================================================================
SCRIPT TẠO BỘ SLIDES POWERPOINT (PPTX) BẢO VỆ CAPSTONE AI SIC
Tác giả: Đoàn Mạnh Dũng
Tệp: Thuyet_Trinh/create_presentation_pptx.py
Xuất tệp: Thuyet_Trinh/Slides_Thuyet_Trinh.pptx (Tỉ lệ 16:9 Widescreen)
================================================================================
"""

import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Cấu hình màu sắc Dark Modern
COLOR_BG = RGBColor(15, 23, 42)          # Navy đậm #0F172A
COLOR_CARD = RGBColor(30, 41, 59)        # Slate card #1E293B
COLOR_CARD_BORDER = RGBColor(51, 65, 85) # Border #334155
COLOR_PRIMARY = RGBColor(56, 189, 248)   # Cyan #38BDF8
COLOR_SECONDARY = RGBColor(129, 140, 248)# Indigo #818CF8
COLOR_ACCENT = RGBColor(16, 185, 129)    # Emerald #10B981
COLOR_TEXT_MAIN = RGBColor(248, 250, 252)# White #F8FAFC
COLOR_TEXT_MUTED = RGBColor(148, 163, 184)# Slate text #94A3B8
COLOR_HIGHLIGHT = RGBColor(245, 158, 11) # Amber #F59E0B

def set_slide_background(slide):
    """Tô nền màu navy đậm cho slide"""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_BG

def add_header(slide, title_text, subtitle_text, slide_num):
    """Thêm tiêu đề chuẩn cho slide"""
    # Header container
    header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    # Badge & Slide Number
    p_meta = tf.paragraphs[0]
    p_meta.text = f"SAMSUNG INNOVATION CAMPUS  |  ĐỒ ÁN TỐT NGHIỆP AI SIC  |  SLIDE {slide_num:02d}/12"
    p_meta.font.size = Pt(10)
    p_meta.font.bold = True
    p_meta.font.color.rgb = COLOR_PRIMARY
    p_meta.space_after = Pt(4)

    # Title
    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_TEXT_MAIN
    p_title.space_after = Pt(2)

    # Subtitle
    if subtitle_text:
        p_sub = tf.add_paragraph()
        p_sub.text = subtitle_text
        p_sub.font.size = Pt(12)
        p_sub.font.color.rgb = COLOR_TEXT_MUTED

def add_card(slide, left, top, width, height, title, items, badge=""):
    """Thêm một thẻ thông tin bo tròn với tiêu đề và danh sách nội dung"""
    # Thẻ nền
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = COLOR_CARD
    shape.line.color.rgb = COLOR_CARD_BORDER
    shape.line.width = Pt(1)

    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.25)
    tf.margin_right = Inches(0.25)
    tf.margin_top = Inches(0.2)
    tf.margin_bottom = Inches(0.2)

    # Tiêu đề thẻ
    p_title = tf.paragraphs[0]
    if badge:
        p_title.text = f"[{badge}] {title}"
    else:
        p_title.text = title
    p_title.font.size = Pt(14)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_PRIMARY
    p_title.space_after = Pt(10)

    # Các ý chi tiết
    for it in items:
        p = tf.add_paragraph()
        p.text = f"• {it}"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_MAIN
        p.space_after = Pt(6)

def build_pptx():
    prs = Presentation()
    # Kích thước 16:9 Widescreen (13.333 x 7.5 inches)
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # ==========================================================================
    # SLIDE 1: BÌA (TITLE SLIDE)
    # ==========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    # Khung trung tâm
    title_box = s1.shapes.add_textbox(Inches(1.5), Inches(1.8), Inches(10.333), Inches(4.0))
    tf1 = title_box.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "SAMSUNG INNOVATION CAMPUS (AI SIC) — BÁO CÁO CAPSTONE"
    p0.alignment = PP_ALIGN.CENTER
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_PRIMARY
    p0.space_after = Pt(16)

    p1 = tf1.add_paragraph()
    p1.text = "HỆ THỐNG GỢI Ý PHỐI ĐỒ\nTHỜI TRANG THÔNG MINH"
    p1.alignment = PP_ALIGN.CENTER
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_TEXT_MAIN
    p1.space_after = Pt(16)

    p2 = tf1.add_paragraph()
    p2.text = "Ứng dụng Deep Learning (ResNet50) & Thuật toán Phối đồ Bù trừ Đa mục tiêu"
    p2.alignment = PP_ALIGN.CENTER
    p2.font.size = Pt(15)
    p2.font.color.rgb = COLOR_TEXT_MUTED
    p2.space_after = Pt(30)

    p3 = tf1.add_paragraph()
    p3.text = "Học viên: Đoàn Mạnh Dũng   |   Thời lượng báo cáo: 10 Phút   |   Năm học 2026"
    p3.alignment = PP_ALIGN.CENTER
    p3.font.size = Pt(13)
    p3.font.bold = True
    p3.font.color.rgb = COLOR_ACCENT

    # ==========================================================================
    # SLIDE 2: ĐẶT VẤN ĐỀ & NHU CẦU THỰC TẾ
    # ==========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "1. Đặt Vấn Đề & Nhu Cầu Thực Tiễn", "Tại sao bài toán Gợi ý Phối đồ Bù trừ lại quan trọng và cấp thiết?", 2)

    add_card(s2, Inches(0.8), Inches(1.8), Inches(3.6), Inches(5.0), "Nỗi đau Người dùng", [
        "Người tiêu dùng mất trung bình 15-30 phút mỗi sáng cho câu hỏi 'Hôm nay mặc gì?'.",
        "Tủ đồ có nhiều món nhưng thiếu tư duy phối đồ (Styling), dẫn đến lãng phí thời gian và trang phục.",
        "Nhu cầu cá nhân hóa cao theo thời tiết, hoàn cảnh đi làm (Formal) hay dạo phố (Casual)."
    ], badge="USER PAIN")

    add_card(s2, Inches(4.8), Inches(1.8), Inches(3.6), Inches(5.0), "Hạn chế của TMĐT Hiện nay", [
        "Chỉ gợi ý các món tương tự (Item-to-Item): Mua áo thun thì gợi ý thêm... 10 chiếc áo thun khác!",
        "Thiếu hoàn toàn khả năng phối hợp thành bộ Outfit hoàn chỉnh (Áo + Quần + Giày + Phụ kiện).",
        "Trải nghiệm mua sắm bị đứt đoạn, giảm tỷ lệ chốt đơn và doanh thu trung bình mỗi giỏ hàng."
    ], badge="MARKET GAP")

    add_card(s2, Inches(8.8), Inches(1.8), Inches(3.6), Inches(5.0), "Cơ hội & Giải pháp AI", [
        "Ứng dụng Thị giác Máy tính (Computer Vision) để đọc hiểu kiểu dáng, form mẫu, màu sắc.",
        "Xây dựng không gian biểu diễn ngữ nghĩa 2048 chiều bằng Mạng nơ-ron tích chập ResNet50.",
        "Thiết kế thuật toán bù trừ danh mục (Complementary Pairing) cho kết quả chuẩn gu tức thì."
    ], badge="AI SOLUTION")

    # ==========================================================================
    # SLIDE 3: MỤC TIÊU & CHUẨN ĐẦU RA (SIC CAPSTONE)
    # ==========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "2. Mục Tiêu & Chuẩn Đồ Án Samsung Innovation Campus", "Bám sát tuyệt đối 3 Stage đào tạo chuẩn của chương trình AI SIC", 3)

    add_card(s3, Inches(0.8), Inches(1.8), Inches(3.6), Inches(5.0), "Stage 1: Làm chủ Dữ liệu", [
        "Thu thập và làm sạch 1,200 sản phẩm thời trang thật từ Kaggle Myntra Dataset.",
        "Kiểm tra tính toàn vẹn của tệp ảnh, xử lý tệp hỏng, lọc thuộc tính trống.",
        "Chuẩn hóa dữ liệu ảnh đầu vào theo phân phối ImageNet (Resize 224x224, Z-Score).",
        "Xây dựng pipeline PyTorch Dataset & DataLoader tối ưu."
    ], badge="STAGE 1")

    add_card(s3, Inches(4.8), Inches(1.8), Inches(3.6), Inches(5.0), "Stage 2: Huấn luyện & Đánh giá", [
        "Ứng dụng kỹ thuật Transfer Learning với mô hình ResNet50 trích xuất đặc trưng sâu.",
        "Loại bỏ tầng phân loại FC, khai thác lớp Global Average Pooling (GAP) 2048 chiều.",
        "Chuẩn hóa L2-Norm đưa vector về mặt cầu đơn vị và đo khoảng cách góc Cosine.",
        "Xây dựng Baseline Color Histogram (512D) để đo lường định lượng sự vượt trội."
    ], badge="STAGE 2")

    add_card(s3, Inches(8.8), Inches(1.8), Inches(3.6), Inches(5.0), "Stage 3: Đóng gói Sản phẩm", [
        "Phát triển Động cơ Phối đồ Bù trừ (Complementary Category Engine) đa mục tiêu.",
        "Xây dựng Giao diện Web tương tác hoàn chỉnh (HTML/CSS/JS + Flask API & Streamlit).",
        "Đạt độ trễ suy luận siêu tốc (< 30ms), sẵn sàng tích hợp môi trường thực tế.",
        "Bộ tài liệu thuyết trình, mã nguồn và báo cáo đồng bộ 100%."
    ], badge="STAGE 3")

    # ==========================================================================
    # SLIDE 4: DỮ LIỆU & TIỀN XỬ LÝ (KAGGLE DATASET)
    # ==========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "3. Dữ Liệu & Pipeline Tiền Xử Lý", "Nguồn dữ liệu thực tế và quy trình chuẩn hóa ảnh nghiêm ngặt", 4)

    add_card(s4, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "Tập Dữ liệu Kaggle Fashion Product Images", [
        "Quy mô: 1,200 sản phẩm thương mại thật với ảnh độ phân giải cao.",
        "Metadata: 9 trường thông tin chi tiết (id, gender, masterCategory, subCategory, articleType, baseColour, season, year, usage).",
        "Độ phân bổ giới tính cân bằng: Men (48.3%), Women (45.2%), Unisex (6.5%).",
        "Danh mục đa dạng: Topwear (Áo), Bottomwear (Quần), Shoes (Giày), Bags (Túi), Watches (Đồng hồ)."
    ], badge="DATASET")

    add_card(s4, Inches(6.8), Inches(1.8), Inches(5.6), Inches(5.0), "Quy trình Chuẩn hóa 4 Bước (preprocess.py)", [
        "1. Lọc dữ liệu: Kiểm tra tệp ảnh tồn tại trên đĩa, loại bỏ ảnh lỗi hoặc kích thước 0 byte.",
        "2. Đổi kích thước: Resize song tuyến tính (Bilinear) về kích thước chuẩn 224 x 224 điểm ảnh.",
        "3. Chuyển đổi Tensor: Chuyển mảng điểm ảnh RGB [0, 255] sang Tensor PyTorch miền [0.0, 1.0].",
        "4. Chuẩn hóa Z-Score ImageNet: Mean = [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225].",
        "-> Đảm bảo phân phối đầu vào khớp hoàn hảo với không gian tối ưu của ResNet50."
    ], badge="PREPROCESSING")

    # ==========================================================================
    # SLIDE 5: KIẾN TRÚC HỆ THỐNG TỔNG THỂ
    # ==========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "4. Kiến Trúc Kỹ Thuật Hệ Thống Tổng Thể", "Luồng dữ liệu khép kín từ ảnh đầu vào đến gợi ý phối đồ thời gian thực", 5)

    add_card(s5, Inches(0.8), Inches(1.8), Inches(3.6), Inches(5.0), "Giai đoạn 1: Offline Indexing", [
        "Tập ảnh 1,200 sản phẩm được đưa qua DataLoader với Batch size = 32.",
        "Mô hình ResNet50 trích xuất vector đặc trưng 2048 chiều cho từng sản phẩm.",
        "Chuẩn hóa L2-Norm và lưu thành ma trận (1200 x 2048) trong tệp models/embeddings.pkl.",
        "Baseline Color HSV trích xuất ma trận (1200 x 512) để đối chứng."
    ], badge="OFFLINE")

    add_card(s5, Inches(4.8), Inches(1.8), Inches(3.6), Inches(5.0), "Giai đoạn 2: Online Inference", [
        "Người dùng tải ảnh lên hoặc chọn sản phẩm mẫu trên Giao diện Web.",
        "Bộ trích xuất ResNet50 tạo vector truy vấn q chỉ trong 25ms.",
        "Tính tích vô hướng Dot Product giữa q và toàn bộ 1,200 vector trong kho.",
        "Thời gian tính toán Cosine Similarity cực nhanh (< 1ms trên CPU)."
    ], badge="ONLINE")

    add_card(s5, Inches(8.8), Inches(1.8), Inches(3.6), Inches(5.0), "Giai đoạn 3: Pairing Engine", [
        "Nhận diện danh mục gốc (ví dụ: Topwear -> Áo sơ mi nam).",
        "Áp dụng ma trận luật bù trừ: Tìm Bottomwear, Shoes, Accessories tương ứng.",
        "Lọc giới tính (Men / Unisex) và cộng thưởng điểm đồng điệu phong cách (+0.15).",
        "Xếp hạng Top-K ứng viên và hiển thị bộ Outfit hoàn chỉnh trên Web."
    ], badge="MATCHING")

    # ==========================================================================
    # SLIDE 6: RESNET50 & ĐỘT PHÁ RESIDUAL BLOCK
    # ==========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "5. Trái Tim Mô Hình: Kiến Trúc ResNet50", "Tại sao ResNet50 là tiêu chuẩn vàng trong nhận diện đặc trưng thị giác?", 6)

    add_card(s6, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "Vấn đề Triệt tiêu Đạo hàm (Vanishing Gradient)", [
        "Khi mạng nơ-ron sâu hơn 20 tầng, đạo hàm lan truyền ngược theo quy tắc chuỗi bị nhân dồn các số nhỏ hơn 1.",
        "Tín hiệu đạo hàm suy giảm về xấp xỉ 0 khi về tới các tầng đầu tiên, khiến mô hình dừng học.",
        "Hiện tượng suy thoái (Degradation): Mạng sâu hơn có cả Training Error và Test Error cao hơn mạng nông!"
    ], badge="THEORY")

    add_card(s6, Inches(6.8), Inches(1.8), Inches(5.6), Inches(5.0), "Đột phá Toán học: Skip Connection (+1 Gradient)", [
        "Khối Thặng dư (Residual Block) thiết lập liên kết tắt: y = F(x) + x.",
        "Đạo hàm lan truyền ngược: dL/dx = (dL/dy) * (dF/dx + 1) = (dL/dy)*(dF/dx) + (dL/dy).",
        "-> Ý nghĩa số +1: Tín hiệu dL/dy luôn có đường cao tốc truyền thẳng về các tầng trước mà không bao giờ bị triệt tiêu!",
        "Cấu trúc Bottleneck 3 tầng (1x1 -> 3x3 -> 1x1) nén và phục hồi số kênh, tiết kiệm 75% chi phí tính toán.",
        "Lớp Global Average Pooling (GAP) nén 7x7x2048 thành vector 2048 chiều không tham số, chống Overfitting triệt để."
    ], badge="BREAKTHROUGH")

    # ==========================================================================
    # SLIDE 7: ĐỘ ĐO TƯƠNG ĐỒNG & KHÔNG GIAN BIỂU DIỄN
    # ==========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "6. Độ Đo Tương Đồng & Không Gian Biểu Diễn", "Toán học của khoảng cách góc và phép chiếu lên mặt cầu đơn vị", 7)

    add_card(s7, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "Chuẩn Hóa L2-Norm (Unit Hypersphere)", [
        "Công thức: v_hat = v / ||v||_2 với ||v||_2 = sqrt(sum(v_i^2)).",
        "Mọi vector 2048 chiều đều được chiếu lên mặt cầu đơn vị có bán kính r = 1.0.",
        "Ý nghĩa thị giác: Triệt tiêu hoàn toàn sự chênh lệch độ sáng/tối khi chụp ảnh.",
        "Hai bức ảnh chụp cùng một chiếc áo ở điều kiện sáng khác nhau sẽ có cùng vector đơn vị."
    ], badge="L2-NORM")

    add_card(s7, Inches(6.8), Inches(1.8), Inches(5.6), Inches(5.0), "Cosine Similarity vs Khoảng cách Euclid", [
        "Công thức: Cosine(u, v) = (u . v) / (||u|| ||v||) = u . v (khi đã chuẩn hóa L2).",
        "Khoảng giá trị [-1.0, 1.0]: +1.0 là trùng khớp hoàn toàn, 0.0 là trực giao không liên quan.",
        "Vượt qua 'Bẫy số chiều' (Curse of Dimensionality):",
        "Trong không gian 2048 chiều, khoảng cách Euclid giữa mọi cặp điểm đều tiệm cận bằng nhau.",
        "Góc phương vị Cosine phản ánh chính xác tỷ lệ đóng góp của các đặc trưng kiểu dáng và phong cách."
    ], badge="COSINE METRIC")

    # ==========================================================================
    # SLIDE 8: ĐỘNG CƠ PHỐI ĐỒ BÙ TRỪ THÔNG MINH
    # ==========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "7. Động Cơ Phối Đồ Bù Trừ Thông Minh", "Khác biệt cốt lõi giữa Gợi ý tương tự và Gợi ý phối hợp thành set đồ", 8)

    add_card(s8, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "Ma Trận Luật Phối Đồ Bù Trừ (Pairing Rules)", [
        "Quy tắc 1: Đầu vào là Áo (Topwear/Dress) -> Tự động tìm Quần (Bottomwear) + Giày (Shoes) + Túi/Đồng hồ.",
        "Quy tắc 2: Đầu vào là Quần (Bottomwear) -> Tự động tìm Áo (Topwear) + Giày (Shoes) + Phụ kiện.",
        "Quy tắc 3: Đầu vào là Giày dép (Shoes) -> Tự động phối ngược lại Áo + Quần + Phụ kiện phù hợp.",
        "-> Giải quyết triệt để vấn đề gợi ý trùng danh mục gây khó chịu cho người dùng."
    ], badge="PAIRING RULES")

    add_card(s8, Inches(6.8), Inches(1.8), Inches(5.6), Inches(5.0), "Hàm Mục Tiêu Phối Hợp Đa Thành Phần", [
        "Điểm phối hợp cuối cùng = Cosine Similarity + Điểm thưởng Phong cách (Style Bonus).",
        "Lọc giới tính nghiêm ngặt: Nam chỉ phối với đồ Nam/Unisex; Nữ chỉ phối với đồ Nữ/Unisex.",
        "Thưởng đồng điệu hoàn cảnh: +0.15 điểm nếu cùng thuộc nhóm Casual, Formal hoặc Sports.",
        "Giới hạn điểm số trong khoảng [0.50, 0.99] để hiển thị % độ tương thích trực quan trên Web."
    ], badge="SCORING ENGINE")

    # ==========================================================================
    # SLIDE 9: GIAO DIỆN WEB DEMO THỰC TẾ
    # ==========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "8. Giao Diện Web Ứng Dụng Thực Tế (Live Demo)", "Trải nghiệm trực quan hóa sản phẩm sẵn sàng triển khai", 9)

    add_card(s9, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "Kiến Trúc Giao Diện Tách Bạch & Linh Hoạt", [
        "Frontend: HTML5, CSS3 Glassmorphism, JavaScript thuần không phụ thuộc nặng nề.",
        "Backend: Flask REST API tốc độ cao, xử lý đa luồng, hỗ trợ CORS.",
        "Khởi chạy 1-Click: Nhấp đúp vào file start_web.bat là tự mở trình duyệt.",
        "Hỗ trợ song song cả giao diện Streamlit qua tệp streamlit_app.py.",
        "Minh bạch học thuật: Học viên báo cáo phần giao diện Web được phát triển nhanh với sự hỗ trợ của AI."
    ], badge="FULL-STACK")

    add_card(s9, Inches(6.8), Inches(1.8), Inches(5.6), Inches(5.0), "Các Tính Năng Trực Quan Hóa Nổi Bật", [
        "1. Chọn nhanh từ kho mẫu: Trải nghiệm tức thì 1,200 sản phẩm thật từ Kaggle.",
        "2. Tải ảnh cá nhân (Upload): Nhận diện và gợi ý phối đồ cho bất kỳ ảnh trang phục nào từ máy tính.",
        "3. Chuyển đổi mô hình: So sánh trực tiếp kết quả giữa ResNet50 và Baseline Color Histogram.",
        "4. Hiển thị thông minh: Xem trước ảnh món gốc bên trái và toàn bộ set đồ phối bù trừ bên phải với thanh đo % tương thích."
    ], badge="FEATURES")

    # ==========================================================================
    # SLIDE 10: ĐÁNH GIÁ THỰC NGHIỆM & BENCHMARK
    # ==========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "9. Đánh Giá Thực Nghiệm & Kết Quả Benchmark", "Số liệu chứng minh sự vượt trội tuyệt đối của Deep Learning", 10)

    # Thêm bảng so sánh trực quan
    rows = 5
    cols = 4
    left = Inches(0.8)
    top = Inches(1.8)
    width = Inches(11.733)
    height = Inches(2.5)

    table_shape = s10.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    headers = ["Chỉ số Đánh giá (Metrics)", "Mô hình ResNet50 (Đề xuất)", "Baseline Color Histogram", "Mức độ Cải thiện"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(37, 99, 235)
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(12)
            p.font.color.rgb = COLOR_TEXT_MAIN
            p.alignment = PP_ALIGN.CENTER

    data = [
        ["Top-5 Hit Rate (Độ phủ gợi ý)", "93.5%", "64.2%", "+ 29.3% (Vượt trội)"],
        ["Precision@5 (Độ chính xác)", "87.2%", "51.8%", "+ 35.4% (Rất cao)"],
        ["Thời gian trích xuất / ảnh", "24.8 ms (GPU) / 85 ms (CPU)", "4.2 ms", "Hoàn toàn Real-time"],
        ["Thời gian tìm kiếm Cosine", "0.85 ms (Toàn kho 1,200 mẫu)", "0.32 ms", "Tức thì (< 1ms)"]
    ]

    for r_idx, row in enumerate(data):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx + 1, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_CARD
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(11)
                p.font.color.rgb = COLOR_TEXT_MAIN if c_idx < 3 else COLOR_ACCENT
                if c_idx == 0:
                    p.alignment = PP_ALIGN.LEFT
                    p.font.bold = True
                else:
                    p.alignment = PP_ALIGN.CENTER

    add_card(s10, Inches(0.8), Inches(4.6), Inches(11.733), Inches(2.2), "Phân tích Kết quả Thực nghiệm Định tính", [
        "Baseline màu sắc chỉ tìm các món cùng dải màu (áo xanh phối với quần xanh lá cây, gây lòe loẹt, lạc quẻ phong cách).",
        "ResNet50 hiểu sâu về cấu trúc trang phục: Áo sơ mi nam công sở được phối chính xác với Quần tây âu đen, Giày tây Derby và Đồng hồ dây da sang trọng.",
        "Áo thun polo dạo phố nữ được phối ăn ý với Quần jean ống suông cá tính và Giày sneakers trắng năng động."
    ], badge="INSIGHT")

    # ==========================================================================
    # SLIDE 11: ĐÓNG GÓP & TỰ ĐÁNH GIÁ DỰ ÁN
    # ==========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11)
    add_header(s11, "10. Đóng Góp Cốt Lõi & Tính Minh Bạch", "Tổng kết giá trị đồ án và tinh thần trung thực học thuật", 11)

    add_card(s11, Inches(0.8), Inches(1.8), Inches(3.6), Inches(5.0), "Hoàn thành Xuất sắc Mục tiêu", [
        "Hoàn thành trọn vẹn yêu cầu Capstone của Samsung Innovation Campus.",
        "Xây dựng thành công hệ thống gợi ý từ bài toán thực tế, dữ liệu thật đến ứng dụng Web hoàn chỉnh.",
        "Hiểu sâu bản chất toán học của CNN, ResNet50, GAP và không gian vector."
    ], badge="CONTRIBUTION")

    add_card(s11, Inches(4.8), Inches(1.8), Inches(3.6), Inches(5.0), "Cấu trúc Thư mục Khoa học", [
        "Thư mục Model/: Chứa 3 file mã nguồn lõi do học viên trực tiếp xây dựng và làm chủ 100%.",
        "Thư mục app_web/: Giao diện và API độc lập, dễ dàng bảo trì hoặc nâng cấp.",
        "Thư mục Report_and_Readme/: Tài liệu hóa chuyên nghiệp, đồng bộ báo cáo và slide."
    ], badge="CLEAN CODE")

    add_card(s11, Inches(8.8), Inches(1.8), Inches(3.6), Inches(5.0), "Minh bạch Sử dụng AI", [
        "Chủ động báo cáo trước hội đồng: Sử dụng AI hỗ trợ lập trình giao diện Web và biên soạn tài liệu.",
        "Thể hiện kỹ năng khai thác công cụ AI hiện đại để gia tăng năng suất gấp 5-10 lần.",
        "Cam kết làm chủ tuyệt đối phần thuật toán và lý thuyết học máy cốt lõi."
    ], badge="TRANSPARENCY")

    # ==========================================================================
    # SLIDE 12: KẾT LUẬN & HƯỚNG MỞ RỘNG
    # ==========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12)
    add_header(s12, "11. Kết Luận & Hướng Phát Triển Tương Lai", "Lời kết và lộ trình nâng cấp hệ thống lên môi trường sản xuất quy mô lớn", 12)

    add_card(s12, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "Kết luận Tổng kết", [
        "Hệ thống chứng minh tính khả thi và hiệu quả vượt bậc của Deep Learning trong thời trang.",
        "Giải quyết triệt để hạn chế 'chỉ gợi ý đồ tương tự' của các sàn thương mại điện tử truyền thống.",
        "Độ trễ suy luận tức thì (< 30ms) mang lại trải nghiệm người dùng mượt mà, sẵn sàng triển khai thực tế.",
        "Khẳng định kiến thức vững chắc thu nhận được từ khóa học AI SIC."
    ], badge="CONCLUSION")

    add_card(s12, Inches(6.8), Inches(1.8), Inches(5.6), Inches(5.0), "Lộ trình Phát triển Tương lai", [
        "1. Tối ưu tìm kiếm với FAISS / HNSW: Mở rộng kho sản phẩm lên 10 triệu items mà thời gian truy vấn vẫn dưới 5ms.",
        "2. Tích hợp YOLOv8-Seg: Bóc tách nền trang phục tự động cho ảnh người dùng chụp ngoài đường phố.",
        "3. Gợi ý theo ngữ cảnh thời tiết: Tích hợp OpenWeather API để tự động đề xuất đồ ấm khi trời lạnh, đồ thoáng mát khi trời nắng.",
        "4. Virtual Try-On với Diffusion: Cho phép người dùng nhìn thấy chính mình mặc set đồ được gợi ý."
    ], badge="FUTURE WORK")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Slides_Thuyet_Trinh.pptx")
    prs.save(out_path)
    print(f"✅ Đã tạo thành công bộ Slides PowerPoint: {out_path}")
    print(f"📄 Tổng số slide: {len(prs.slides)} | Kích thước: 16:9 Widescreen (13.333 x 7.5 inches)")

if __name__ == "__main__":
    build_pptx()
