"""
================================================================================
HỆ THỐNG GỢI Ý PHỐI ĐỒ THỜI TRANG THÔNG MINH (AI SIC CAPSTONE)
MÁY CHỦ WEB API BACKEND (FLASK SERVER)
Tệp: app_web/backend.py
(Mô-đun được xây dựng với sự hỗ trợ của AI nhằm phục vụ Demo Web)
================================================================================
"""

import os
import sys
import io
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from model_caller import get_model_caller

app = Flask(
    __name__,
    template_folder=os.path.join(CURRENT_DIR, "templates"),
    static_folder=os.path.join(CURRENT_DIR, "static")
)

DATASET_IMAGES_DIR = os.path.join(PROJECT_ROOT, "dataset", "images")

@app.route("/")
def index():
    """Trang chủ giao diện web"""
    return render_template("index.html")

@app.route("/images/<filename>")
def serve_dataset_image(filename):
    """Phục vụ ảnh sản phẩm thời trang từ thư mục dataset/images/"""
    if os.path.exists(os.path.join(DATASET_IMAGES_DIR, filename)):
        return send_from_directory(DATASET_IMAGES_DIR, filename)
    # Fallback placeholder nếu không thấy
    return "", 404

@app.route("/api/sample_items", methods=["GET"])
def get_sample_items():
    """Lấy danh sách các sản phẩm mẫu nổi bật trong kho để người dùng chọn nhanh"""
    caller = get_model_caller()
    df = caller.df_metadata

    sample_items = []
    # Lấy các mẫu đại diện đa dạng: Topwear, Bottomwear, Shoes, Dresses
    categories = ['Topwear', 'Bottomwear', 'Shoes', 'Dress']
    for cat in categories:
        sub_df = df[df['subCategory'] == cat]
        for _, row in sub_df.head(4).iterrows():
            idx = int(row.name)
            sample_items.append({
                "index": idx,
                "id": int(row['id']),
                "name": str(row['productDisplayName']),
                "subCategory": str(row['subCategory']),
                "gender": str(row['gender']),
                "usage": str(row['usage']),
                "baseColour": str(row['baseColour']),
                "image_url": f"/images/{row['id']}.jpg"
            })

    return jsonify({"items": sample_items})

@app.route("/api/recommend", methods=["POST"])
def recommend_api():
    """
    API Nhận yêu cầu gợi ý phối đồ:
    - Nếu gửi JSON: chứa query_idx
    - Nếu gửi Form data: chứa file ảnh upload
    """
    caller = get_model_caller()

    use_baseline = False
    preferred_gender = None
    preferred_usage = None

    if request.is_json:
        data = request.get_json()
        query_idx = data.get("query_idx")
        use_baseline = bool(data.get("use_baseline", False))
        preferred_gender = data.get("preferred_gender")
        preferred_usage = data.get("preferred_usage")

        result = caller.recommend_outfit(
            query_idx=query_idx,
            use_baseline=use_baseline,
            preferred_gender=preferred_gender,
            preferred_usage=preferred_usage
        )
    else:
        # Nhận file upload
        if "image" not in request.files:
            return jsonify({"error": "Không tìm thấy file ảnh tải lên!"}), 400

        file = request.files["image"]
        use_baseline = request.form.get("use_baseline", "false").lower() == "true"
        preferred_gender = request.form.get("preferred_gender")
        preferred_usage = request.form.get("preferred_usage")

        try:
            image_bytes = file.read()
            pil_img = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return jsonify({"error": f"Tệp ảnh không hợp lệ: {str(e)}"}), 400

        query_vector = caller.extract_vector_from_image(pil_img, use_baseline=use_baseline)

        result = caller.recommend_outfit(
            query_vector=query_vector,
            use_baseline=use_baseline,
            preferred_gender=preferred_gender,
            preferred_usage=preferred_usage
        )

    # Chuyển đổi đường dẫn ảnh sang URL thân thiện với trình duyệt
    if result["origin"].get("image_path"):
        img_id = os.path.splitext(os.path.basename(result["origin"]["image_path"]))[0]
        result["origin"]["image_url"] = f"/images/{img_id}.jpg"

    for cat, items in result["recommendations"].items():
        for item in items:
            item["image_url"] = f"/images/{item['id']}.jpg"

    return jsonify(result)

def start_server(host="127.0.0.1", port=5000, debug=False):
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    start_server(debug=True)
