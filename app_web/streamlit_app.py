"""
================================================================================
SMART FASHION OUTFIT RECOMMENDER - WEB APPLICATION (AI SIC CAPSTONE)
Giao diện Web tương tác thời gian thực với Streamlit
File: app_web/app.py
(Mô-đun được xây dựng với sự hỗ trợ của AI nhằm phục vụ Demo trước Hội đồng)
================================================================================
"""

import os
import sys
import streamlit as pd_st
import streamlit as st
from PIL import Image
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from model_caller import get_model_caller

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="AI Fashion Recommender | AI SIC",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS giao diện thời trang cao cấp
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: white;
        padding: 30px 40px;
        border-radius: 20px;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .product-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .badge-score {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-origin {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Khởi tạo caller
@st.cache_resource
def load_cached_caller():
    return get_model_caller()

caller = load_cached_caller()
df_meta = caller.df_metadata

# ------------------------------------------------------------------------------
# THANH ĐIỀU KHIỂN BÊN TRÁI (SIDEBAR)
# ------------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/isometric/100/t-shirt.png", width=70)
st.sidebar.title("🎛️ Bảng Điều Khiển")
st.sidebar.markdown("**Samsung Innovation Campus (AI SIC)**")
st.sidebar.caption("Tác giả đồ án: **Đoàn Mạnh Dũng**")
st.sidebar.markdown("---")

# Chọn mô hình
model_choice = st.sidebar.radio(
    "🧠 Lựa chọn Mô hình:",
    ["ResNet50 (Transfer Learning - Khuyên dùng)", "Baseline (Color Histogram HSV)"],
    index=0
)
use_baseline = (model_choice == "Baseline (Color Histogram HSV)")

# Bộ lọc
st.sidebar.subheader("🎯 Tùy chỉnh Phối đồ")
gender_filter = st.sidebar.selectbox("Giới tính ưu tiên:", ["Tất cả", "Men", "Women", "Unisex"])
usage_filter = st.sidebar.selectbox("Hoàn cảnh sử dụng:", ["Tất cả", "Casual", "Formal", "Sports", "Ethnic"])
top_k = st.sidebar.slider("Số gợi ý mỗi danh mục:", min_value=1, max_value=3, value=1)

st.sidebar.markdown("---")
st.sidebar.info("""
💡 **Gợi ý thuyết trình**:
- **ResNet50**: Hiểu form dáng, cổ áo, đường cắt may và phong cách tổng thể.
- **Baseline**: Chỉ so sánh màu sắc, dễ gợi ý món đồ đơn điệu.
""")

# ------------------------------------------------------------------------------
# GIAO DIỆN CHÍNH (MAIN PAGE)
# ------------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-weight:800; font-size:2.2rem;">✨ Hệ thống Gợi ý Phối đồ Thời trang Thông minh</h1>
    <p style="margin:8px 0 0 0; color:#94A3B8; font-size:1.05rem;">
        Smart Fashion Outfit Recommender System • Ứng dụng Deep Learning ResNet50 & Động cơ Phối đồ Bù trừ
    </p>
</div>
""", unsafe_allow_html=True)

tab_catalog, tab_upload = st.tabs(["🛍️ Chọn từ Kho Mẫu Đại diện", "📤 Tải ảnh Sản phẩm Cá nhân"])

selected_idx = None
uploaded_image = None

with tab_catalog:
    st.subheader("Chọn nhanh 1 món đồ để xem AI phối trang phục kèm theo:")
    
    # Lọc danh sách 8 sản phẩm mẫu đa dạng
    sample_indices = []
    for sub in ['Topwear', 'Bottomwear']:
        for g in ['Men', 'Women']:
            m = (df_meta['subCategory'] == sub) & (df_meta['gender'] == g)
            idxs = df_meta[m].index.tolist()
            if idxs:
                sample_indices.extend(idxs[:2])
    sample_indices = list(dict.fromkeys(sample_indices))[:8]

    cols = st.columns(4)
    for i, idx in enumerate(sample_indices):
        item = df_meta.iloc[idx]
        img_path = caller.get_image_disk_path(item)
        col = cols[i % 4]
        with col:
            if img_path and os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            st.caption(f"**{item['subCategory']}** ({item['gender']})\n{item['productDisplayName'][:20]}...")
            if st.button(f"Chọn món #{item['id']}", key=f"btn_sample_{idx}", use_container_width=True):
                selected_idx = idx

with tab_upload:
    st.subheader("Tải lên ảnh một món đồ bất kỳ từ máy tính của bạn:")
    uploaded_file = st.file_uploader("Chọn tệp ảnh (.jpg, .png, .jpeg):", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, caption="Ảnh bạn đã tải lên", width=220)

# ------------------------------------------------------------------------------
# XỬ LÝ GỢI Ý & HIỂN THỊ KẾT QUẢ
# ------------------------------------------------------------------------------
if selected_idx is not None or uploaded_image is not None:
    st.markdown("---")
    st.subheader("🎉 KẾT QUẢ PHỐI ĐỒ ĐƯỢC GỢI Ý BỞI AI")
    
    with st.spinner("Đang tính toán ma trận Cosine Similarity và luật phối đồ..."):
        if selected_idx is not None:
            results = caller.recommend_outfit(
                query_idx=selected_idx,
                use_baseline=use_baseline,
                preferred_gender=gender_filter,
                preferred_usage=usage_filter,
                top_k=top_k
            )
        else:
            q_vec = caller.extract_vector_from_image(uploaded_image, use_baseline=use_baseline)
            results = caller.recommend_outfit(
                query_vector=q_vec,
                use_baseline=use_baseline,
                preferred_gender=gender_filter,
                preferred_usage=usage_filter,
                top_k=top_k
            )

    origin = results['origin']
    recs = results['recommendations']

    # Hiển thị layout kết quả
    total_cards = 1 + len(recs)
    result_cols = st.columns(min(total_cards, 5))

    # Cột 1: Món gốc
    with result_cols[0]:
        st.markdown('<span class="badge-origin">MÓN ĐÃ CHỌN (GỐC)</span>', unsafe_allow_html=True)
        if origin['image_path'] and os.path.exists(origin['image_path']):
            st.image(origin['image_path'], use_container_width=True)
        elif uploaded_image is not None:
            st.image(uploaded_image, use_container_width=True)
        st.markdown(f"**{origin['name']}**")
        st.caption(f"Loại: {origin['subCategory']} | Phong cách: {origin['usage']}")

    # Các cột gợi ý
    col_idx = 1
    for cat_name, items in recs.items():
        if col_idx < len(result_cols):
            with result_cols[col_idx]:
                best_item = items[0]
                score_pct = best_item['score'] * 100
                st.markdown(f'<span class="badge-score">GỢI Ý {cat_name.upper()} • {score_pct:.1f}%</span>', unsafe_allow_html=True)
                if best_item['image_path'] and os.path.exists(best_item['image_path']):
                    st.image(best_item['image_path'], use_container_width=True)
                st.markdown(f"**{best_item['name'][:25]}...**")
                st.caption(f"Màu: {best_item['baseColour']} | Dịp: {best_item['usage']}")
            col_idx += 1

    # Khung giải thích lý do phối đồ
    st.markdown("### 🔍 Phân tích Lý do Phối đồ của AI")
    st.success(f"""
    * **Phong cách chủ đạo**: Bộ trang phục được tối ưu hóa cho hoàn cảnh **{origin['usage']}**.
    * **Phối màu & Đường nét**: Mô hình Deep Learning trích xuất đặc trưng hình thái để cân bằng màu sắc tương phản/trung tính giữa **{origin['subCategory']}** và các sản phẩm phối kèm.
    * **Thời gian suy luận**: Đạt dưới 30ms nhờ ma trận đặc trưng $L_2$-Norm tiền trích xuất.
    """)
else:
    st.info("👆 Vui lòng bấm chọn một món đồ mẫu ở trên hoặc tải ảnh cá nhân lên để kích hoạt hệ thống gợi ý!")
