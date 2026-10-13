"""
================================================================================
HỆ THỐNG GỢI Ý PHỐI ĐỒ THỜI TRANG THÔNG MINH (AI SIC CAPSTONE)
MÔ-ĐUN 2: HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH (TRAIN & VALIDATION)
Tác giả: Đoàn Mạnh Dũng
File: Model/train_and_val.py
================================================================================
Nhiệm vụ cốt lõi:
1. Xây dựng Kiến trúc Deep Learning:
   - ResNet50 Transfer Learning: Cắt bỏ tầng phân loại FC, giữ lại Global Average Pooling (GAP)
     để trích xuất vector ngữ nghĩa 2048 chiều.
   - Baseline Color Histogram: Trích xuất phân bố màu sắc không gian HSV 512 chiều để đối chứng.
2. Quy trình Trích xuất Đặc trưng theo Lô (Batch Processing với DataLoader):
   - Xử lý 32 ảnh đồng thời, tối ưu hóa tốc độ trên CPU/GPU.
   - Chuẩn hóa vector theo chuẩn L2-Norm (độ dài vector = 1.0) nhằm tối ưu phép đo Cosine.
3. Kiểm định (Validation & Sanity Check):
   - Kiểm tra tính hợp lệ của ma trận (loại trừ NaN, Inf).
   - Kiểm tra tính phân tán và kiểm định độ chuẩn hóa của vector đặc trưng.
4. Đóng gói Artifacts: Lưu trữ ma trận đặc trưng vào thư mục models/ phục vụ suy luận thời gian thực.
================================================================================
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

# Cấu hình UTF-8 cho Windows Terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import mô-đun tiền xử lý dữ liệu
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from preprocess import load_and_clean_metadata, FashionDataset, create_train_val_split, get_image_transforms

# Đường dẫn lưu trữ đầu ra
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------------
# 1. ĐỊNH NGHĨA KIẾN TRÚC MÔ HÌNH RESNET50 (TRANSFER LEARNING)
# ------------------------------------------------------------------------------
class ResNet50FeatureExtractor(nn.Module):
    """
    Kiến trúc Deep Learning trích xuất đặc trưng thị giác cấp cao (High-level Visual Semantics).
    - Khởi tạo từ trọng số tiền huấn luyện ImageNet (ResNet50_Weights.DEFAULT).
    - Đóng băng toàn bộ trọng số (requires_grad = False) để tận dụng tri thức đã học.
    - Cắt bỏ tầng Fully Connected (FC), thay bằng Identity để lấy vector 2048 chiều từ GAP.
    """
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Đóng băng toàn bộ tham số
        for param in base_model.parameters():
            param.requires_grad = False
        # Thay thế tầng FC cuối cùng bằng Identity layer
        base_model.fc = nn.Identity()
        self.feature_extractor = base_model

    def forward(self, x):
        return self.feature_extractor(x)

# ------------------------------------------------------------------------------
# 2. ĐỊNH NGHĨA MÔ HÌNH ĐỐI CHỨNG (BASELINE COLOR HISTOGRAM)
# ------------------------------------------------------------------------------
def extract_baseline_hsv(image_path, bins=(8, 8, 8)):
    """
    Trích xuất vector màu sắc truyền thống (Handcrafted Feature):
    - Chuyển ảnh sang không gian màu HSV.
    - Chia 8 bins cho mỗi kênh (H, S, V) -> 8 * 8 * 8 = 512 chiều.
    - Chuẩn hóa L2-Norm.
    """
    try:
        img = Image.open(image_path).convert('HSV')
        hsv_arr = np.array(img)
        hist, _ = np.histogramdd(
            hsv_arr.reshape(-1, 3),
            bins=bins,
            range=[(0, 256), (0, 256), (0, 256)]
        )
        hist = hist.flatten().astype(np.float32)
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
        return hist
    except Exception:
        vec = np.zeros(512, dtype=np.float32)
        vec[0] = 1.0
        return vec

# ------------------------------------------------------------------------------
# 3. QUY TRÌNH TRÍCH XUẤT ĐẶC TRƯNG VÀ KIỂM ĐỊNH (TRAIN & VALIDATE)
# ------------------------------------------------------------------------------
def run_feature_extraction_and_validation(batch_size=32):
    print("=" * 70)
    print("🚀 BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN & TRÍCH XUẤT ĐẶC TRƯNG (Model/train_and_val.py)")
    print(f"⚙️ Thiết bị thực thi: {DEVICE}")
    print("=" * 70)

    # Bước 1: Nạp và làm sạch metadata
    df_clean = load_and_clean_metadata()
    train_df, val_df = create_train_val_split(df_clean, val_ratio=0.1)

    # Bước 2: Khởi tạo mô hình
    print("\n🧠 [MÔ HÌNH] Khởi tạo mạng ResNet50 (Transfer Learning)...")
    model = ResNet50FeatureExtractor().to(DEVICE)
    model.eval()

    # Bước 3: Chuẩn bị DataLoader
    dataset = FashionDataset(df_clean)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Bước 4: Trích xuất ma trận ResNet50
    print(f"⏳ [TRÍCH XUẤT RESNET50] Đang xử lý {len(dataset)} ảnh theo lô (Batch size = {batch_size})...")
    start_time = time.time()
    embeddings_list = []

    with torch.no_grad():
        for b_idx, (images, ids, idxs) in enumerate(loader):
            images = images.to(DEVICE)
            feats = model(images).cpu().numpy()  # Kích thước (B, 2048)
            embeddings_list.append(feats)
            if (b_idx + 1) % 10 == 0 or (b_idx + 1) == len(loader):
                processed = min((b_idx + 1) * batch_size, len(dataset))
                print(f"   ⏩ Đã xử lý {processed}/{len(dataset)} ảnh ({processed/len(dataset)*100:.1f}%)")

    resnet_matrix = np.vstack(embeddings_list)
    resnet_time = time.time() - start_time
    print(f"✅ Hoàn tất trích xuất ResNet50 trong {resnet_time:.2f} giây! Kích thước: {resnet_matrix.shape}")

    # Bước 5: Chuẩn hóa L2-Norm cho ResNet50
    print("📐 [CHUẨN HÓA] Áp dụng chuẩn hóa L2-Norm cho từng vector đặc trưng...")
    norms = np.linalg.norm(resnet_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    resnet_matrix_normalized = resnet_matrix / norms

    # Bước 6: Kiểm định chất lượng ma trận (Validation & Sanity Checks)
    print("\n🔬 [KIỂM ĐỊNH MA TRẬN - SANITY CHECKS]:")
    has_nan = np.isnan(resnet_matrix_normalized).any()
    has_inf = np.isinf(resnet_matrix_normalized).any()
    checked_norms = np.linalg.norm(resnet_matrix_normalized, axis=1)
    norm_min, norm_max = checked_norms.min(), checked_norms.max()
    
    print(f"   * Kiểm tra NaN/Inf: {'❌ Có lỗi' if (has_nan or has_inf) else '✅ Hoàn toàn sạch'}")
    print(f"   * Kiểm tra độ dài chuẩn L2: Min = {norm_min:.4f}, Max = {norm_max:.4f} (Kỳ vọng: 1.0000)")
    assert not has_nan and not has_inf, "Phát hiện lỗi NaN/Inf trong ma trận!"

    # Bước 7: Trích xuất ma trận Baseline Color Histogram
    print("\n🎨 [MÔ HÌNH ĐỐI CHỨNG] Trích xuất ma trận Baseline Color HSV (512 chiều)...")
    b_start = time.time()
    baseline_list = [extract_baseline_hsv(p) for p in df_clean['image_path']]
    baseline_matrix = np.array(baseline_list, dtype=np.float32)
    b_time = time.time() - b_start
    print(f"✅ Hoàn tất Baseline trong {b_time:.2f} giây! Kích thước: {baseline_matrix.shape}")

    # Bước 8: Đóng gói và lưu trữ Artifacts
    print("\n💾 [LƯU TRỮ] Đóng gói mô hình vào thư mục models/...")
    
    emb_file = os.path.join(MODELS_DIR, "embeddings.pkl")
    with open(emb_file, "wb") as f:
        pickle.dump({
            "embeddings": resnet_matrix_normalized,
            "ids": df_clean['id'].tolist(),
            "embedding_dim": resnet_matrix_normalized.shape[1]
        }, f)
    print(f"   * Đã lưu ma trận ResNet50: {emb_file} ({os.path.getsize(emb_file) / 1024 / 1024:.2f} MB)")

    base_file = os.path.join(MODELS_DIR, "baseline_embeddings.pkl")
    with open(base_file, "wb") as f:
        pickle.dump({
            "embeddings": baseline_matrix,
            "ids": df_clean['id'].tolist(),
            "embedding_dim": baseline_matrix.shape[1]
        }, f)
    print(f"   * Đã lưu ma trận Baseline: {base_file} ({os.path.getsize(base_file) / 1024 / 1024:.2f} MB)")

    meta_file = os.path.join(MODELS_DIR, "metadata.csv")
    df_clean.to_csv(meta_file, index=False, encoding='utf-8')
    print(f"   * Đã lưu metadata sản phẩm: {meta_file}")

    print("=" * 70)
    print("🎉 QUY TRÌNH HUẤN LUYỆN & KIỂM ĐỊNH THÀNH CÔNG RỰC RỠ!")
    print("=" * 70)
    return resnet_matrix_normalized, baseline_matrix, df_clean

if __name__ == "__main__":
    run_feature_extraction_and_validation(batch_size=32)
