"""
================================================================================
HỆ THỐNG GỢI Ý PHỐI ĐỒ THỜI TRANG THÔNG MINH (AI SIC CAPSTONE)
MÔ-ĐUN GỌI MÔ HÌNH & SUY LUẬN THỜI GIAN THỰC (MODEL CALLER & INFERENCE ENGINE)
Thư mục: app_web/model_caller.py
(Mô-đun được xây dựng với sự hỗ trợ của AI nhằm phục vụ kết nối Web & Model)
================================================================================
Nhiệm vụ cốt lõi:
1. Nạp và quản lý bộ nhớ đệm (Cache) cho ma trận Embeddings và Metadata:
   - ResNet50 Embeddings (1200 x 2048)
   - Baseline HSV Embeddings (1200 x 512)
   - Metadata sản phẩm
2. Trích xuất đặc trưng tức thì cho ảnh người dùng tải lên từ Web (Inference).
3. Động cơ Phối đồ Bù trừ (Complementary Category Engine):
   - Nhận diện loại đồ gốc (Topwear, Bottomwear, Shoes...).
   - Tính toán Cosine Similarity trên toàn kho sản phẩm.
   - Lọc theo giới tính và tính điểm hòa hợp phong cách (Style consistency score).
================================================================================
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# Xác định đường dẫn gốc dự án
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform chuẩn ImageNet
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------------------------------------------------------
# 1. QUẢN LÝ TẢI TRỌNG SỐ VÀ METADATA (CACHED MODEL CALLER)
# ------------------------------------------------------------------------------
class ModelCaller:
    """
    Lớp điều phối gọi mô hình và phục vụ suy luận cho Giao diện Web.
    """
    def __init__(self):
        self.device = DEVICE
        self.resnet_extractor = None
        self.resnet_matrix = None
        self.baseline_matrix = None
        self.df_metadata = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Nạp ma trận embeddings và bảng metadata"""
        emb_path = os.path.join(MODELS_DIR, "embeddings.pkl")
        base_path = os.path.join(MODELS_DIR, "baseline_embeddings.pkl")
        meta_path = os.path.join(MODELS_DIR, "metadata.csv")

        # Nạp ResNet50 Embeddings
        if os.path.exists(emb_path):
            with open(emb_path, "rb") as f:
                data = pickle.load(f)
            self.resnet_matrix = data['embeddings']
        else:
            raise FileNotFoundError(f"Không tìm thấy ma trận ResNet50 tại {emb_path}")

        # Nạp Baseline Embeddings
        if os.path.exists(base_path):
            with open(base_path, "rb") as f:
                b_data = pickle.load(f)
            self.baseline_matrix = b_data['embeddings']

        # Nạp Metadata
        if os.path.exists(meta_path):
            self.df_metadata = pd.read_csv(meta_path)
        else:
            raise FileNotFoundError(f"Không tìm thấy metadata tại {meta_path}")

    def get_resnet_extractor(self):
        """Khởi tạo lười (Lazy init) mạng trích xuất ResNet50 để tiết kiệm RAM"""
        if self.resnet_extractor is None:
            base = resnet50(weights=ResNet50_Weights.DEFAULT)
            for p in base.parameters():
                p.requires_grad = False
            base.fc = nn.Identity()
            self.resnet_extractor = base.to(self.device)
            self.resnet_extractor.eval()
        return self.resnet_extractor

    def extract_vector_from_image(self, pil_image, use_baseline=False):
        """Trích xuất vector đặc trưng từ một ảnh PIL tùy ý"""
        if use_baseline:
            # Baseline: HSV Color Histogram
            hsv = pil_image.convert('HSV')
            hsv_arr = np.array(hsv)
            hist, _ = np.histogramdd(hsv_arr.reshape(-1, 3), bins=(8, 8, 8), range=[(0, 256), (0, 256), (0, 256)])
            vec = hist.flatten().astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            return vec.reshape(1, -1)
        else:
            # ResNet50 Deep Feature Extractor
            extractor = self.get_resnet_extractor()
            img_tensor = INFERENCE_TRANSFORMS(pil_image.convert('RGB')).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = extractor(img_tensor).cpu().numpy().flatten()
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat /= norm
            return feat.reshape(1, -1)

    def get_image_disk_path(self, item_row):
        """Lấy đường dẫn tệp ảnh hợp lệ trên ổ cứng"""
        if 'image_path' in item_row and pd.notna(item_row['image_path']) and os.path.exists(str(item_row['image_path'])):
            return str(item_row['image_path'])
        item_id = str(item_row['id'])
        for folder in ["dataset", "data"]:
            p = os.path.join(PROJECT_ROOT, folder, "images", f"{item_id}.jpg")
            if os.path.exists(p):
                return p
        return None

    def get_pairing_rules(self, sub_category):
        """Luật phối đồ bù trừ thời trang"""
        if sub_category in ['Topwear', 'Dress']:
            return ['Bottomwear', 'Shoes', 'Watches', 'Bags']
        elif sub_category in ['Bottomwear']:
            return ['Topwear', 'Shoes', 'Watches', 'Bags']
        elif sub_category in ['Shoes', 'Sandals', 'Flip Flops']:
            return ['Topwear', 'Bottomwear', 'Watches', 'Bags']
        else:
            return ['Topwear', 'Bottomwear', 'Shoes']

    def recommend_outfit(self, query_vector=None, query_idx=None, use_baseline=False, 
                         preferred_gender=None, preferred_usage=None, top_k=1):
        """
        Hàm gọi mô hình suy luận hoàn chỉnh phục vụ Web:
        - Nhận vào query_vector (ảnh upload) hoặc query_idx (chọn từ kho mẫu).
        - Trả về thông tin món gốc và danh sách gợi ý phối kèm.
        """
        matrix = self.baseline_matrix if use_baseline else self.resnet_matrix

        if query_idx is not None:
            query_row = self.df_metadata.iloc[query_idx]
            q_vec = matrix[query_idx:query_idx+1]
            detected_sub = query_row['subCategory']
            detected_gender = query_row['gender']
            detected_usage = query_row['usage']
            origin_name = query_row['productDisplayName']
            origin_img_path = self.get_image_disk_path(query_row)
        else:
            q_vec = query_vector
            # Tính độ tương đồng với toàn kho để nhận diện thuộc tính sản phẩm upload
            sims_all = np.dot(matrix, q_vec.T).flatten()
            best_match_idx = int(np.argmax(sims_all))
            query_row = self.df_metadata.iloc[best_match_idx]
            detected_sub = query_row['subCategory']
            detected_gender = query_row['gender']
            detected_usage = query_row['usage']
            origin_name = f"Nhận diện: {query_row['productDisplayName']}"
            origin_img_path = None

        # Thiết lập bộ lọc
        gender = preferred_gender if preferred_gender and preferred_gender != "Tất cả" else detected_gender
        usage = preferred_usage if preferred_usage and preferred_usage != "Tất cả" else detected_usage

        # Tính Cosine Similarity với toàn kho
        all_scores = np.dot(matrix, q_vec.T).flatten()

        # Áp dụng luật phối đồ bù trừ
        target_cats = self.get_pairing_rules(detected_sub)
        outfit_recs = {}

        for cat in target_cats:
            mask = (
                (self.df_metadata['subCategory'] == cat) &
                (self.df_metadata['gender'].isin([gender, 'Unisex']))
            )
            if query_idx is not None:
                mask = mask & (self.df_metadata.index != query_idx)

            cand_indices = self.df_metadata[mask].index.tolist()
            if not cand_indices:
                continue

            candidates = []
            for c_idx in cand_indices:
                row_c = self.df_metadata.iloc[c_idx]
                raw_sim = float(all_scores[c_idx])
                # Thưởng điểm phong cách nếu đồng điệu hoàn cảnh sử dụng
                bonus = 0.15 if row_c['usage'] == usage else 0.0
                final_score = float(min(0.99, max(0.50, raw_sim + bonus)))
                
                img_path = self.get_image_disk_path(row_c)
                candidates.append({
                    "id": int(row_c['id']),
                    "name": str(row_c['productDisplayName']),
                    "subCategory": str(row_c['subCategory']),
                    "baseColour": str(row_c['baseColour']),
                    "usage": str(row_c['usage']),
                    "score": final_score,
                    "image_path": img_path
                })

            candidates.sort(key=lambda x: x['score'], reverse=True)
            if candidates:
                outfit_recs[cat] = candidates[:top_k]

        return {
            "origin": {
                "name": origin_name,
                "subCategory": detected_sub,
                "gender": gender,
                "usage": usage,
                "image_path": origin_img_path
            },
            "recommendations": outfit_recs
        }

# Khởi tạo singleton phục vụ import nhanh
_caller_instance = None
def get_model_caller():
    global _caller_instance
    if _caller_instance is None:
        _caller_instance = ModelCaller()
    return _caller_instance
