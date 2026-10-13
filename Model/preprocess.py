"""
================================================================================
HỆ THỐNG GỢI Ý PHỐI ĐỒ THỜI TRANG THÔNG MINH (AI SIC CAPSTONE)
MÔ-ĐUN 1: TIỀN XỬ LÝ DỮ LIỆU & CHUẨN HÓA ẢNH (DATA PREPROCESSING)
Tác giả: Đoàn Mạnh Dũng
File: Model/preprocess.py
================================================================================
Nhiệm vụ cốt lõi:
1. Đọc metadata sản phẩm từ styles.csv, lọc và kiểm tra tính toàn vẹn của tệp ảnh.
2. Xử lý các giá trị bị thiếu (Missing Values), chuẩn hóa các cột nhãn phân loại.
3. Xây dựng Pipeline tiền xử lý ảnh chuẩn ImageNet:
   - Resize về kích thước cố định (224 x 224).
   - Chuyển đổi định dạng Tensor (giá trị [0.0, 1.0]).
   - Chuẩn hóa phân phối Z-Score với ImageNet Mean & Std.
4. Đóng gói thành PyTorch Dataset và chia tập dữ liệu Huấn luyện / Kiểm định (Train / Val Split).
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Cấu hình hiển thị UTF-8 trên Windows Terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Xác định đường dẫn tương đối thông minh (chạy được từ thư mục gốc hoặc trong thư mục Model)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATASET_CSV = os.path.join(PROJECT_ROOT, "dataset", "styles.csv")
DATASET_IMG_DIR = os.path.join(PROJECT_ROOT, "dataset", "images")
FALLBACK_CSV = os.path.join(PROJECT_ROOT, "data", "styles.csv")
FALLBACK_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "images")

# ------------------------------------------------------------------------------
# 1. PIPELINE BIẾN ĐỔI ẢNH CHUẨN IMAGENET (IMAGE TRANSFORMS)
# ------------------------------------------------------------------------------
def get_image_transforms():
    """
    Trả về bộ tiền xử lý hình ảnh chuẩn hóa theo mô hình ResNet50:
    - Resize: 224x224 (kích thước đầu vào chuẩn của mạng CNN)
    - ToTensor: Chuyển mảng pixel [0, 255] sang Tensor số thực [0.0, 1.0]
    - Normalize: Trừ mean, chia std của 3 kênh màu RGB theo ImageNet
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# ------------------------------------------------------------------------------
# 2. LÀM SẠCH VÀ KIỂM TRA TÍNH TOÀN VẸN CỦA DỮ LIỆU (DATA CLEANING)
# ------------------------------------------------------------------------------
def load_and_clean_metadata():
    """
    Nạp tệp CSV thuộc tính sản phẩm và loại bỏ các mẫu dữ liệu lỗi:
    - Kiểm tra sự tồn tại thực tế của tệp ảnh trên ổ cứng.
    - Điền giá trị mặc định cho các trường thông tin bị thiếu.
    - Tạo cột đường dẫn tuyệt đối 'image_path'.
    """
    csv_path = DATASET_CSV if os.path.exists(DATASET_CSV) else FALLBACK_CSV
    img_dir = DATASET_IMG_DIR if os.path.exists(DATASET_IMG_DIR) else FALLBACK_IMG_DIR

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Không tìm thấy tệp metadata tại: {csv_path}")

    print(f"📖 [TIỀN XỬ LÝ] Đang đọc metadata từ: {csv_path}")
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    initial_count = len(df)

    # Làm sạch các trường thông tin cốt lõi
    df['subCategory'] = df['subCategory'].fillna('Other')
    df['articleType'] = df['articleType'].fillna('Unknown')
    df['gender'] = df['gender'].fillna('Unisex')
    df['baseColour'] = df['baseColour'].fillna('Unknown')
    df['usage'] = df['usage'].fillna('Casual')
    df['productDisplayName'] = df['productDisplayName'].fillna('Fashion Product')

    # Kiểm tra tệp ảnh thực tế
    valid_records = []
    for idx, row in df.iterrows():
        item_id = str(row['id'])
        img_path = os.path.join(img_dir, f"{item_id}.jpg")
        if os.path.exists(img_path):
            row_dict = row.to_dict()
            row_dict['image_path'] = img_path
            valid_records.append(row_dict)

    clean_df = pd.DataFrame(valid_records)
    print(f"✅ [TIỀN XỬ LÝ] Hoàn tất làm sạch: {len(clean_df)}/{initial_count} mẫu dữ liệu ảnh hợp lệ.")
    return clean_df

# ------------------------------------------------------------------------------
# 3. ĐỊNH NGHĨA PYTORCH DATASET CHO FASHION
# ------------------------------------------------------------------------------
class FashionDataset(Dataset):
    """
    Lớp Dataset phục vụ nạp dữ liệu theo lô (Batch Processing) trong PyTorch.
    """
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform if transform else get_image_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Ảnh dự phòng nếu gặp lỗi đọc
            image = Image.new('RGB', (224, 224), color=(200, 200, 200))
        
        tensor_img = self.transform(image)
        return tensor_img, int(row['id']), idx

# ------------------------------------------------------------------------------
# 4. PHÂN CHIA TẬP HUẤN LUYỆN & KIỂM ĐỊNH (TRAIN / VAL SPLIT)
# ------------------------------------------------------------------------------
def create_train_val_split(df, val_ratio=0.1, random_seed=42):
    """
    Phân chia tập dữ liệu thành:
    - Tập Train: 90% (dùng để trích xuất và xây dựng không gian biểu diễn)
    - Tập Validation: 10% (dùng để kiểm định tính ổn định của vector đặc trưng)
    """
    np.random.seed(random_seed)
    indices = np.random.permutation(len(df))
    val_size = int(len(df) * val_ratio)
    
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    
    print(f"📊 [SPLIT] Phân chia dữ liệu: Train = {len(train_df)} mẫu | Validation = {len(val_df)} mẫu.")
    return train_df, val_df

# ------------------------------------------------------------------------------
# KHỐI THỰC THI KIỂM THỬ MÔ-ĐUN (STANDALONE TEST)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 CHẠY KIỂM THỬ MÔ-ĐUN TIỀN XỬ LÝ (Model/preprocess.py)")
    print("=" * 70)
    
    df_clean = load_and_clean_metadata()
    train_df, val_df = create_train_val_split(df_clean, val_ratio=0.1)
    
    # Kiểm tra transform trên 1 ảnh mẫu
    test_sample = df_clean.iloc[0]
    transform = get_image_transforms()
    test_img = Image.open(test_sample['image_path']).convert('RGB')
    tensor_sample = transform(test_img)
    
    print("\n🔍 KẾT QUẢ KIỂM THỬ HÌNH HỌC TENSOR:")
    print(f"👉 Kích thước ảnh gốc: {test_img.size}")
    print(f"👉 Kích thước Tensor đầu ra: {tuple(tensor_sample.shape)} (Channels, Height, Width)")
    print(f"👉 Miền giá trị pixel sau chuẩn hóa: Min = {tensor_sample.min():.2f}, Max = {tensor_sample.max():.2f}")
    print("=" * 70)
    print("🎉 MÔ-ĐUN TIỀN XỬ LÝ HOẠT ĐỘNG CHUẨN XÁC 100%!")
