"""
================================================================================
SCRIPT TỰ ĐỘNG XUẤT BÁO CÁO & CẨM NANG KỸ THUẬT PDF (CHUẨN SAMSUNG INNOVATION CAMPUS)
Tác giả: Đoàn Mạnh Dũng
Vị trí: Report_and_Readme/build_all_documents_and_pdfs.py
================================================================================
TÍNH NĂNG VƯỢT TRỘI ĐÃ NÂNG CẤP:
1. SỬA TRIỆT ĐỂ LỖI CÔNG THỨC TOÁN:
   - Tự động chuyển đổi công thức LaTeX ($...$ và $$...$$) sang chuẩn W3C MathML
     qua thư viện latex2mathml.
   - Chrome Headless in ấn MathML trực tiếp với font Cambria Math / Latin Modern Math,
     hiển thị sắc nét 100%, không mất dấu, không lỗi font.
2. NÂNG CẤP KIẾN THỨC TOÀN DIỆN (TỪ SỐ 0 ĐẾN CHUYÊN GIA):
   - Giải thích bản chất CNN, Receptive Field, Padding, Stride, Activation (ReLU/Sigmoid).
   - Batch Normalization, GAP Pooling, ResNet50 Bottleneck, Chứng minh toán học Skip Connection.
   - Không gian vector đa chiều, L2-Norm, Cosine Similarity, Bẫy số chiều (Curse of Dimensionality).
   - Lý thuyết Recommender Systems, Complementary Category Matching, Độ đo đánh giá Top-K Hit Rate, Precision@K, MAP@K.
   - Kỹ thuật mở rộng quy mô công nghiệp (FAISS, HNSW, IVF-PQ).
3. ĐỒNG BỘ CẤU TRÚC THƯ MỤC MỚI NHẤT:
   - Thư mục Thuyet_Trinh/: Có cả Slides_Thuyet_Trinh.pptx và Slides_Thuyet_Trinh.pdf (16:9).
   - Thư mục app_web/: Tách bạch templates/index.html, static/css, static/js, backend.py, run_web.py, start_web.bat.
   - Thư mục Model/: 3 file code lõi (preprocess.py, train_and_val.py, test_model.py) + notebook.
   - Thư mục gốc Root: Chỉ để duy nhất README.md, README AI SIC.pdf và requirements.txt.
================================================================================
"""

import os
import sys
import re
import shutil
import subprocess
import bs4
import markdown
import pymupdf
import latex2mathml.converter

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURRENT_DIR) == "Report_and_Readme":
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    DOCS_DIR = CURRENT_DIR
else:
    PROJECT_ROOT = CURRENT_DIR
    DOCS_DIR = os.path.join(PROJECT_ROOT, "Report_and_Readme")

os.makedirs(DOCS_DIR, exist_ok=True)

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
BROWSER_EXE = CHROME_PATH if os.path.exists(CHROME_PATH) else EDGE_PATH

# ==============================================================================
# BỘ XỬ LÝ CÔNG THỨC TOÁN HỌC LATEX SANG MATHML CHUẨN
# ==============================================================================
def sanitize_latex_expr(expr):
    """Làm sạch các biểu thức LaTeX đặc biệt trước khi đưa vào converter"""
    expr = expr.strip()
    # Xử lý các lệnh đặc thù
    expr = expr.replace(r'\mathbb{I}', r'\mathbf{I}')
    expr = expr.replace(r'\dots', r'\ldots')
    return expr

def convert_latex_to_mathml_in_text(md_text):
    """
    Tìm kiếm và thay thế các công thức LaTeX $$...$$ và $...$ thành thẻ MathML
    giúp Chrome Headless in ra PDF hiển thị công thức toán học chuẩn quốc tế.
    """
    def replace_block(match):
        raw = match.group(1).strip()
        sanitized = sanitize_latex_expr(raw)
        try:
            m = latex2mathml.converter.convert(sanitized, display="block")
            return f"\n\n<div class=\"math-block\">{m}</div>\n\n"
        except Exception:
            return f"\n\n<div class=\"math-block\"><pre class=\"math-raw\">{raw}</pre></div>\n\n"

    def replace_inline(match):
        raw = match.group(1).strip()
        sanitized = sanitize_latex_expr(raw)
        try:
            m = latex2mathml.converter.convert(sanitized, display="inline")
            return f"<span class=\"math-inline\">{m}</span>"
        except Exception:
            return f"<code class=\"math-raw-inline\">{raw}</code>"

    # Xử lý Block Math $$...$$ trước
    text = re.sub(r'\$\$(.*?)\$\$', replace_block, md_text, flags=re.DOTALL)
    # Xử lý Inline Math $...$ (không phải dấu $$)
    text = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', replace_inline, text)
    return text

# ==============================================================================
# PHẦN KIẾN THỨC NỀN TẢNG CHUYÊN SÂU (TỪ SỐ 0 ĐẾN CHUYÊN GIA)
# ==============================================================================
FOUNDATIONAL_KNOWLEDGE = r"""
### 1.1 Sự Tiến Hóa của Trí Tuệ Nhân Tạo và Thị Giác Máy Tính
Để thấu hiểu bài toán từ con số 0, ta cần nắm rõ tiến trình phát triển của ngành khoa học này:
* **Hệ thống dựa trên luật (Rule-Based Systems - 1970s)**: Con người lập trình thủ công toàn bộ các tập luật `if-else`. Nhược điểm chí mạng là không thể bao quát được thế giới thị giác với vô vàn góc chụp, nếp nhăn và ánh sáng thay đổi.
* **Học máy cổ điển (Classical Machine Learning - 1990s - 2000s)**: Sử dụng các thuật toán như SVM (Support Vector Machine), Random Forest hay k-NN kết hợp với các bộ trích xuất đặc trưng thủ công (Hand-crafted Features) như **SIFT**, **HOG (Histogram of Oriented Gradients)** hay **Color Histogram**. Nhược điểm: Đặc trưng thủ công phụ thuộc nặng nề vào kinh nghiệm chuyên gia, kém bền vững khi góc nhìn hoặc tỷ lệ vật thể thay đổi.
* **Học sâu (Deep Learning - 2012 đến nay)**: Đột phá với sự ra đời của AlexNet trên tập dữ liệu ImageNet. Đặc điểm cốt lõi là **Tự động học đặc trưng phân tầng (Hierarchical End-to-End Representation Learning)**: Các tầng đầu học nét vẽ, viền cạnh; các tầng giữa học họa tiết, chất liệu; các tầng sâu học toàn bộ kiểu dáng trang phục mà không cần bất kỳ sự can thiệp thủ công nào.

---

### 1.2 Mạng Nơ-ron Tích Chập (CNN) và Cơ Sở Toán Học
Hình ảnh số trong máy tính là một Tensor 3 chiều có kích thước $H \times W \times C$ (Chiều cao $\times$ Chiều rộng $\times$ Số kênh màu RGB). Mạng nơ-ron truyền thống (MLP) kéo phẳng ma trận thành một vector dài, làm phá vỡ hoàn toàn cấu trúc không gian 2D lân cận. **CNN ra đời nhằm khai thác tối đa tương quan không gian này**.

#### 1.2.1 Phép Tích Chập 2D (2D Convolution) và Cơ Chế Bộ Lọc (Kernel/Filter)
* **Khái niệm**: Phép tích chập là thao tác trượt một ma trận trọng số nhỏ gọi là **Hạt nhân (Kernel / Filter)** có kích thước $K \times K$ (thường là $3 \times 3$) qua toàn bộ bức ảnh từ trái qua phải, từ trên xuống dưới.
* **Công thức toán học đầy đủ**:
  Tại mỗi tọa độ không gian $(i, j)$, giá trị đầu ra của bản đồ đặc trưng (Feature Map) $S(i, j)$ được tính theo công thức:
  $$S(i, j) = (I * K)(i, j) = \sum_{m} \sum_{n} I(i + m, j + n) \cdot K(m, n) + b$$
  *Trong đó*:
  * $I(i + m, j + n)$ là giá trị cường độ điểm ảnh tại vùng lân cận cục bộ.
  * $K(m, n)$ là các trọng số học được của hạt nhân tích chập.
  * $b$ là hệ số điều chỉnh (bias).
* **Hai đặc tính toán học mang tính cách mạng của CNN**:
  1. **Chia sẻ trọng số (Weight / Parameter Sharing)**: Toàn bộ bức ảnh dùng chung một bộ trọng số hạt nhân $K$. Nhờ đó, thay vì hàng triệu tham số như mạng nơ-ron truyền thống, một lớp tích chập chỉ cần $3 \times 3 \times C_{\text{in}} \times C_{\text{out}}$ tham số, giảm thiểu nguy cơ Overfitting.
  2. **Tính tương đương tịnh tiến (Translation Equivariance)**: Dù chiếc túi xách hay cổ áo nằm ở góc trên bên trái hay ở giữa khung hình, bộ lọc tích chập vẫn sẽ kích hoạt và nhận diện được đặc trưng đó.

#### 1.2.2 Vùng Cảm Nhận Thị Giác (Receptive Field)
* **Khái niệm**: Receptive Field là kích thước vùng không gian trên ảnh đầu vào mà một nơ-ron ở tầng sâu có thể "nhìn thấy".
* **Công thức mở rộng**: Khi xếp chồng hai lớp tích chập $3 \times 3$, vùng cảm nhận đạt $5 \times 5$; xếp chồng ba lớp $3 \times 3$ đạt $7 \times 7$.
* **Ý nghĩa kiến trúc**: Thay vì dùng một bộ lọc lớn $7 \times 7$ (tốn $49$ tham số), mạng ResNet xếp chồng ba bộ lọc $3 \times 3$ (chỉ tốn $3 \times (3 \times 3) = 27$ tham số, tiết kiệm 45%), vừa mở rộng vùng nhìn toàn cục vừa bổ sung thêm 3 lần hàm kích hoạt phi tuyến tính giúp mạng học được nhiều biểu diễn phức tạp hơn.

#### 1.2.3 Đệm Viền (Zero Padding) và Bước Nhảy (Stride)
* **Công thức tính kích thước bản đồ đặc trưng đầu ra**:
  $$O = \left\lfloor \frac{I - K + 2P}{S} \right\rfloor + 1$$
  *Trong đó*: $I$ là kích thước đầu vào, $K$ là kích thước hạt nhân, $P$ là số pixel đệm viền (Padding), $S$ là bước nhảy (Stride).
* **Same Padding ($P = 1, S = 1, K = 3$)**: Đảm bảo kích thước đầu ra $O = I$. Giúp bảo toàn thông tin ở các mép viền ngoài cùng của trang phục (cạp quần, đường may gấu áo) không bị teo nhỏ dần qua các tầng.

#### 1.2.4 Hàm Kích Hoạt Phi Tuyến Tính (Activation Functions)
* **Tại sao bắt buộc phải có tính phi tuyến?** Nếu không có hàm kích hoạt, việc xếp chồng $N$ tầng tích chập chỉ tương đương với một phép nhân ma trận tuyến tính duy nhất: $W_2(W_1 x + b_1) + b_2 = W_{\text{new}} x + b_{\text{new}}$, mạng không thể học được các quan hệ phi tuyến phức tạp của thời trang.
* **Hàm Sigmoid và Hạn chế**:
  $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
  *Nhược điểm*: Đạo hàm cực đại của Sigmoid chỉ đạt $0.25$ ($\sigma'(x) \le 0.25$). Khi mạng có hàng chục tầng, nhân liên tiếp các số nhỏ hơn $0.25$ làm đạo hàm suy giảm về $0$, gây tê liệt quá trình huấn luyện.
* **Hàm ReLU (Rectified Linear Unit - Chuẩn mực hiện đại)**:
  $$f(x) = \max(0, x)$$
  *Ý nghĩa toán học và thị giác*:
  1. Đạo hàm khi $x > 0$ luôn bằng $1.0$, giải phóng hoàn toàn dòng chảy đạo hàm trong mạng sâu.
  2. Tạo tính thưa (Sparsity): Tắt các nơ-ron có giá trị âm về $0$, giúp mạng tập trung năng lực vào các nơ-ron nhận diện đặc trưng có giá trị cao.

#### 1.2.5 Lớp Chuẩn Hóa Theo Lô (Batch Normalization - BN)
* **Hiện tượng Trôi dạt Phân phối Nội bộ (Internal Covariate Shift)**: Khi trọng số các tầng trước thay đổi, phân phối dữ liệu đầu vào của các tầng sau liên tục bị xáo trộn.
* **Công thức toán học của Batch Normalization**:
  $$\mu_B = \frac{1}{m} \sum_{i=1}^m x_i, \quad \sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2$$
  $$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$
  *Trong đó*: $\gamma$ và $\beta$ là hai tham số học được (scale và shift).
* **Lợi ích**: Giúp mô hình hội tụ nhanh hơn gấp 5 lần, cho phép sử dụng Learning Rate cao hơn và đóng vai trò như một bộ điều chuẩn ngăn ngừa hiện tượng học vẹt.

#### 1.2.6 Lớp Gộp: Max Pooling vs. Global Average Pooling (GAP)
* **Max Pooling ($2 \times 2$)**: Lấy giá trị lớn nhất trong từng cửa sổ $2 \times 2$, giảm kích thước không gian đi $50\%$, giữ lại tín hiệu biên cạnh mạnh nhất.
* **Global Average Pooling (GAP - Đột phá của ResNet)**:
  Tính giá trị trung bình số học của toàn bộ điểm ảnh trên từng kênh bản đồ đặc trưng:
  $$\text{GAP}(c) = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} X_{i, j, c}$$
  *Ý nghĩa*: Biến đổi bản đồ đặc trưng $7 \times 7 \times 2048$ thành vector **2048 chiều** duy nhất. Hoàn toàn không sinh thêm bất kỳ tham số nào, loại bỏ hoàn toàn tầng Fully Connected cổ điển (vốn chiếm hơn $80\%$ tham số và là ổ Overfitting của mô hình VGGNet cũ).

---

### 1.3 Kiến Trúc Đột Phá ResNet50 và Khối Thặng Dư (Residual Block)

#### 1.3.1 Vấn Đề Suy Thoái Mô Hình (Degradation Problem)
Nghiên cứu của Kaiming He et al. (CVPR 2016) chỉ ra rằng: Khi tăng độ sâu của mạng nơ-ron từ 20 tầng lên 56 tầng, sai số trên tập huấn luyện (Training Error) và kiểm thử (Test Error) **đều tăng lên**. Đây không phải là hiện tượng Overfitting (vì nếu Overfitting thì training error phải rất thấp), mà là do cấu trúc mạng sâu thông thường làm tê liệt quá trình truyền đạo hàm.

#### 1.3.2 Chứng Minh Toán Học: Cơ Chế Skip Connection (+1 Gradient)
Thay vì bắt các tầng tích chập học ánh xạ toàn phần $\mathcal{H}(x)$, ResNet bắt các tầng học **ánh xạ thặng dư** $\mathcal{F}(x) = \mathcal{H}(x) - x$. Đầu ra của khối thặng dư được cộng trực tiếp với đầu vào $x$ thông qua đường liên kết tắt (Identity Shortcut):
$$y = \mathcal{F}(x, \{W_i\}) + x$$

Áp dụng quy tắc chuỗi để tính đạo hàm của hàm mất mát $\mathcal{L}$ theo đầu vào $x$:
$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left( \frac{\partial \mathcal{F}}{\partial x} + 1 \right) = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial \mathcal{F}}{\partial x} + \frac{\partial \mathcal{L}}{\partial y}$$

> **Ý nghĩa của số $+1$**: Dù tích chập $\frac{\partial \mathcal{F}}{\partial x}$ có suy giảm về $0$, thành phần đạo hàm $\frac{\partial \mathcal{L}}{\partial y}$ vẫn luôn có một "siêu cao tốc" truyền trực tiếp về các tầng trước mà không bao giờ bị nhân triệt tiêu! Đây chính là bí quyết giúp mạng ResNet mở rộng lên 50, 101, thậm chí 152 tầng hoạt động hoàn hảo.

#### 1.3.3 Thiết Kế Khối Bottleneck 3 Tầng
Để tối ưu hóa chi phí tính toán cho mạng 50 tầng, ResNet50 sử dụng cấu trúc Bottleneck:
1. **Conv $1 \times 1$**: Nén số kênh từ $256$ xuống $64$ (giảm chiều dữ liệu).
2. **Conv $3 \times 3$**: Trích xuất đặc trưng không gian ở không gian số kênh thấp ($64$).
3. **Conv $1 \times 1$**: Khôi phục và mở rộng số kênh từ $64$ lên $256$ (hoặc lên $2048$ ở stage cuối).
*So sánh*: Thiết kế này tiết kiệm hơn $75\%$ khối lượng tính toán (FLOPs) so với việc dùng 2 lớp $3 \times 3$ có cùng số kênh $256$.

---

### 1.4 Phương Pháp Học Chuyển Giao (Transfer Learning)
* **Khái niệm**: Huấn luyện một mạng nơ-ron từ đầu (Scratch) trên tập dữ liệu thời trang nhỏ đòi hỏi hàng chục ngàn ảnh gán nhãn và hàng trăm giờ GPU. Học chuyển giao cho phép tái sử dụng các tri thức thị giác đã học được từ một bài toán lớn sang bài toán mới.
* **Tập dữ liệu ImageNet**: Gồm hơn $1.28$ triệu bức ảnh tự nhiên thuộc $1,000$ danh mục khác nhau. Mô hình ResNet50 tiền huấn luyện trên ImageNet đã tích lũy khả năng nhận diện hình thái, đường cong, chất liệu và màu sắc ở mức siêu việt.
* **Chiến lược Bộ trích xuất đặc trưng cố định (Fixed Feature Extractor)**:
  Ta đóng băng (Freeze) toàn bộ trọng số của 50 tầng tích chập và lớp GAP, loại bỏ lớp Softmax 1,000 danh mục ImageNet và lấy thẳng vector 2048 chiều làm biểu diễn đặc trưng thời trang.
  *Ưu điểm*: Tốc độ trích xuất cực nhanh, không tốn tài nguyên huấn luyện lại và hoàn toàn miễn nhiễm với hiện tượng Overfitting trên tập dữ liệu thời trang cục bộ.

---

### 1.5 Không Gian Vector Đặc Trưng và Hình Học Độ Đo Tương Đồng

#### 1.5.1 Vector Đặc Trưng (Feature Embeddings)
Vector đặc trưng là một điểm tọa độ trong không gian số thực 2048 chiều $\mathbb{R}^{2048}$. Mỗi chiều đại diện cho một thuộc tính trừu tượng của sản phẩm: form dáng áo, độ dài tay áo, chất liệu vải jean, cổ bẻ hay cổ tròn, gam màu nóng hay lạnh. Hai sản phẩm tương đồng về thị giác sẽ nằm gần nhau trong không gian này.

#### 1.5.2 Phép Chuẩn Hóa $L_2$-Norm (Đưa về Siêu Cầu Đơn Vị)
Độ dài Euclid của vector $v = (v_1, v_2, \dots, v_D)$:
$$\|v\|_2 = \sqrt{\sum_{i=1}^{D} v_i^2}$$

Phép chuẩn hóa đưa vector về độ dài bằng $1.0$:
$$\hat{v} = \frac{v}{\|v\|_2} \implies \|\hat{v}\|_2 = 1.0$$

*Ý nghĩa thị giác cốt lõi*: Triệt tiêu ảnh hưởng của độ sáng chênh lệch trong ảnh chụp. Hai bức ảnh chụp cùng một chiếc áo sơ mi — một ảnh chụp trong studio sáng rực rỡ và một ảnh chụp thiếu sáng — sẽ có vector cùng hướng nhưng độ dài khác nhau. Phép chuẩn hóa $L_2$ đưa cả hai về cùng một điểm duy nhất trên mặt cầu đơn vị.

#### 1.5.3 Độ Tương Đồng Cosine (Cosine Similarity)
Được định nghĩa là cosin của góc $\theta$ giữa hai vector đặc trưng:
$$\text{Cosine}(u, v) = \cos(\theta) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = \frac{\sum_{i=1}^D u_i v_i}{\sqrt{\sum_{i=1}^D u_i^2} \sqrt{\sum_{i=1}^D v_i^2}}$$

Khi cả hai vector đã được chuẩn hóa $L_2$ ($\|u\|_2 = 1$ và $\|v\|_2 = 1$), công thức suy biến thành **Tích vô hướng (Dot Product)**:
$$\text{Cosine}(u, v) = u \cdot v = \sum_{i=1}^{D} u_i v_i$$

*Mối liên hệ toán học với khoảng cách Euclid bình phương*:
$$\|u - v\|_2^2 = \|u\|_2^2 + \|v\|_2^2 - 2 (u \cdot v) = 1 + 1 - 2 \cos(\theta) = 2(1 - \cos(\theta))$$
Công thức trên chứng minh: **Cặp vector có khoảng cách Euclid càng nhỏ thì độ tương đồng Cosine càng lớn**.

#### 1.5.4 Tại Sao Cosine Similarity Vượt Trội Hơn Khoảng Cách Euclid?
* **Bẫy số chiều (Curse of Dimensionality)**: Trong không gian 2048 chiều, hiện tượng tập trung khoảng cách (Distance Concentration Effect) xảy ra:
  $$\lim_{D \to \infty} \frac{\text{dist}_{\max} - \text{dist}_{\min}}{\text{dist}_{\min}} \to 0$$
  Khoảng cách Euclid giữa mọi cặp điểm đều tiệm cận bằng nhau, làm mất khả năng phân biệt.
* **Góc phương vị phản ánh ngữ nghĩa**: Góc $\theta$ đo lường tỷ lệ phân bổ năng lượng giữa các nơ-ron nhận diện thuộc tính chứ không đo độ lớn biên độ điểm ảnh. Do đó, Cosine Similarity phản ánh độ hòa hợp phong cách chính xác hơn nhiều so với khoảng cách số học tuyệt đối.

---

### 1.6 Lý Thuyết Hệ Thống Gợi Ý (Recommender Systems) và Phối Đồ Bù Trừ

#### 1.6.1 Phân Biệt Các Trường Phái Gợi Ý
* **Lọc cộng tác (Collaborative Filtering)**: Dự đoán dựa trên lịch sử tương tác của người dùng khác (User-Item Interaction Matrix). *Hạn chế*: Gặp vấn đề Khởi động nguội (Cold-Start Problem) nghiêm trọng với các mặt hàng thời trang mới ra mắt chưa có lượt mua.
* **Lọc dựa trên nội dung (Content-Based Filtering)**: Phân tích trực tiếp thuộc tính và hình ảnh của sản phẩm. Không phụ thuộc vào lịch sử người dùng, giải quyết triệt để bài toán Cold-Start.

#### 1.6.2 Khác Biệt Cốt Lõi: Gợi Ý Tương Tự vs Gợi Ý Bù Trừ
* **Gợi ý Tương tự (Item-to-Item Similarity)**: Tìm sản phẩm thay thế (Substitutes). Nếu người dùng xem áo phông đỏ, hệ thống gợi ý các áo phông đỏ khác. Điều này hữu ích cho so sánh giá nhưng vô dụng khi người dùng muốn biết mặc áo này với quần gì!
* **Gợi ý Bù trừ (Complementary Category Recommendation)**: Tìm sản phẩm phối hợp (Complements). Khi người dùng chọn áo phông, hệ thống gợi ý Quần jean, Giày thể thao và Balo đồng điệu để tạo thành một set đồ hoàn chỉnh (Complete Outfit).

#### 1.6.3 Hàm Chấm Điểm Phối Đồ Đa Mục Tiêu (Multi-Objective Utility Function)
Điểm hòa hợp phong cách giữa món đồ truy vấn $q$ và món đồ ứng viên $c$:
$$\text{Score}(q, c) = w_1 \cdot \text{Cosine}(v_q, v_c) + w_2 \cdot \mathbb{I}_{(\text{usage}_q = \text{usage}_c)} + b$$
*Trong đó*: $\mathbb{I}$ là hàm chỉ thị nhận giá trị $1$ nếu cùng hoàn cảnh sử dụng (cùng Casual hoặc cùng Formal) và $0$ nếu khác hoàn cảnh.

---

### 1.7 Các Độ Đo Đánh Giá Hệ Thống Gợi Ý (Evaluation Metrics)
1. **Top-K Hit Rate ($HR@K$)**: Tỷ lệ các truy vấn mà trong Top K sản phẩm được gợi ý có ít nhất một món đồ phối hợp chính xác theo nhãn kiểm định:
   $$HR@K = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}(\text{Hits}(q, K) \ge 1)$$
2. **Precision@K ($P@K$)**: Tỷ lệ phần trăm các sản phẩm được đề xuất trong Top K là thực sự phù hợp:
   $$P@K = \frac{|\text{Relevant Items} \cap \text{Top-K}|}{K}$$
3. **Mean Average Precision (MAP@K)**: Đánh giá độ chính xác có tính đến thứ tự xếp hạng (sản phẩm phù hợp nhất xuất hiện càng ở vị trí đầu thì điểm càng cao).

---

### 1.8 Mở Rộng Quy Mô Công Nghiệp (Production Scalability)
* **Giới hạn của k-NN Vét cạn (Brute-force KNN)**: Độ phức tạp tính toán là $\mathcal{O}(N \cdot D)$. Khi kho hàng có $10^7$ sản phẩm, một truy vấn tốn hàng tỷ phép tính tích vô hướng, gây nghẽn hệ thống.
* **Giải pháp Tìm kiếm Láng giềng Gần đúng (Approximate Nearest Neighbors - ANN)**:
  * **HNSW (Hierarchical Navigable Small World)**: Cấu trúc đồ thị phân tầng đa tỷ lệ, đạt tốc độ tìm kiếm $\mathcal{O}(\log N)$.
  * **IVF-PQ (Inverted File with Product Quantization)**: Nén vector 2048 chiều float32 (8KB) xuống còn 64 bytes thông qua phép lượng tử hóa tích, tăng tốc độ truy vấn lên $1000 \times$.
  * **Công cụ chuẩn công nghiệp**: Meta FAISS, Google ScaNN, Milvus Vector Database.
"""

# ==============================================================================
# HÀM TẠO HTML VỚI MATHML VÀ LIÊN KẾT MỤC LỤC HOẠT ĐỘNG CHUẨN XÁC TRONG PDF
# ==============================================================================
def make_html_with_working_links(md_text, cover_type="report"):
    # 1. Chuyển đổi công thức LaTeX sang MathML
    md_with_mathml = convert_latex_to_mathml_in_text(md_text)

    # 2. Parse Markdown sang HTML
    md_parser = markdown.Markdown(extensions=['tables', 'fenced_code'])
    html_body = md_parser.convert(md_with_mathml)

    soup = bs4.BeautifulSoup(html_body, 'html.parser')

    # 3. Gán ID duy nhất cho các Heading để mục lục nhảy trang hoạt động 100%
    headings = []
    for el in soup.find_all(['h1', 'h2', 'h3']):
        if "MỤC LỤC TỔNG QUAN" in el.text.upper():
            continue
        headings.append(el)

    heading_lookup = {}
    for idx, h in enumerate(headings):
        sec_id = f"sec-target-{idx+1}"
        h['id'] = sec_id
        clean_text = re.sub(r'[^a-zA-Z0-9\u00C0-\u1EF9]', '', h.text.lower())
        heading_lookup[clean_text] = sec_id

    for a in soup.find_all('a'):
        raw_text = a.text
        clean_link = re.sub(r'[^a-zA-Z0-9\u00C0-\u1EF9]', '', raw_text.lower())
        matched_id = None
        for h_text, h_id in heading_lookup.items():
            if clean_link == h_text or clean_link in h_text or h_text in clean_link:
                matched_id = h_id
                break
        if matched_id:
            a['href'] = f"#{matched_id}"

    html_body_final = str(soup)

    if cover_type == "report":
        cover_html = """
        <div class="cover-page">
            <div class="cover-title">BÁO CÁO</div>
            <div class="cover-author">Đoàn Mạnh Dũng</div>
        </div>
        """
        doc_title = "Report AI SIC - Đoàn Mạnh Dũng"
    else:
        cover_html = """
        <div class="cover-page">
            <div class="cover-title">README</div>
        </div>
        """
        doc_title = "README - Hệ thống Gợi ý Phối đồ Thời trang Thông minh"

    html_template = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>__DOC_TITLE__</title>
    <style>
        @page {
            size: A4;
            margin: 20mm 18mm 20mm 18mm;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 10.5pt;
            line-height: 1.6;
            color: #1E293B;
            margin: 0;
            padding: 0;
        }
        .cover-page {
            page-break-after: always;
            height: 90vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
        }
        .cover-title {
            font-size: 44pt;
            font-weight: 900;
            letter-spacing: 4px;
            color: #0F172A;
            margin-bottom: 25px;
        }
        .cover-author {
            font-size: 22pt;
            font-weight: 600;
            color: #2563EB;
            letter-spacing: 1px;
        }
        .page-break {
            page-break-before: always;
        }
        h1 {
            font-size: 18pt;
            color: #0F172A;
            border-bottom: 2px solid #2563EB;
            padding-bottom: 6px;
            margin-top: 30px;
            margin-bottom: 15px;
            page-break-after: avoid;
        }
        h2 {
            font-size: 14pt;
            color: #1E3A8A;
            margin-top: 22px;
            margin-bottom: 10px;
            page-break-after: avoid;
        }
        h3 {
            font-size: 11.5pt;
            color: #047857;
            margin-top: 16px;
            margin-bottom: 8px;
            page-break-after: avoid;
        }
        h4 {
            font-size: 10.5pt;
            color: #B45309;
            margin-top: 12px;
            margin-bottom: 6px;
            page-break-after: avoid;
        }
        p, li {
            text-align: justify;
        }
        a {
            color: #2563EB;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        blockquote {
            border-left: 4px solid #3B82F6;
            margin: 14px 0;
            padding: 10px 16px;
            background-color: #EFF6FF;
            color: #1E3A8A;
            border-radius: 0 8px 8px 0;
            page-break-inside: avoid;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            font-size: 9.5pt;
            page-break-inside: avoid;
        }
        th, td {
            border: 1px solid #CBD5E1;
            padding: 8px 10px;
            text-align: left;
        }
        th {
            background-color: #F1F5F9;
            color: #0F172A;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #F8FAFC;
        }
        code {
            background-color: #F1F5F9;
            padding: 2px 5px;
            border-radius: 4px;
            font-family: Consolas, "Courier New", monospace;
            font-size: 9.5pt;
            color: #BE185D;
        }
        pre {
            background-color: #0F172A;
            color: #F8FAFC;
            padding: 12px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 9pt;
            line-height: 1.45;
            page-break-inside: avoid;
        }
        pre code {
            background-color: transparent;
            color: #F8FAFC;
            padding: 0;
        }
        /* MATHML STYLING CHO CÔNG THỨC TOÁN SẮC NÉT */
        math {
            font-family: "Cambria Math", "Latin Modern Math", STIXGeneral, serif;
            font-size: 1.15em;
        }
        .math-block {
            display: block;
            text-align: center;
            margin: 14px auto;
            padding: 10px 14px;
            background: #F8FAFC;
            border-radius: 8px;
            border: 1px solid #E2E8F0;
            page-break-inside: avoid;
            overflow-x: auto;
        }
        .math-inline {
            display: inline-block;
            padding: 0 2px;
        }
        .math-raw, .math-raw-inline {
            color: #1E3A8A;
            font-family: Consolas, monospace;
        }
        ul, ol {
            padding-left: 22px;
        }
        li {
            margin-bottom: 4px;
        }
        hr {
            border: none;
            border-top: 1px solid #E2E8F0;
            margin: 20px 0;
        }
    </style>
</head>
<body>

__COVER_HTML__

<div class="content-body">
__CONTENT_HTML__
</div>

</body>
</html>
"""
    full_html = html_template.replace("__DOC_TITLE__", doc_title)
    full_html = full_html.replace("__COVER_HTML__", cover_html)
    full_html = full_html.replace("__CONTENT_HTML__", html_body_final)
    return full_html

def convert_html_to_pdf(html_file, pdf_file):
    print(f"🖨️ Đang xuất file PDF qua Chrome Headless: {pdf_file}...")
    abs_html = os.path.abspath(html_file)
    abs_pdf = os.path.abspath(pdf_file)

    cmd = [
        BROWSER_EXE,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={abs_pdf}",
        abs_html
    ]
    subprocess.run(cmd, check=True)

    # Kiểm tra liên kết bằng PyMuPDF
    doc = pymupdf.open(abs_pdf)
    total_links = sum(len(page.get_links()) for page in doc)
    print(f"📄 File: {pdf_file} | Tổng số trang: {len(doc)} | Tổng liên kết nội bộ (Clickable): {total_links}")

# ==============================================================================
# 1. BIÊN SOẠN BÁO CÁO REPORT.MD & REPORT AI SIC.PDF TRONG REPORT_AND_README/
# ==============================================================================
def generate_report():
    print("\n[1/2] Đang tạo report.md và Report AI SIC.pdf trong Report_and_Readme/...")

    report_p1 = r"""# BÁO CÁO

---

## MỤC LỤC TỔNG QUAN

- [PHẦN 1: TỔNG QUAN KIẾN THỨC NỀN TẢNG — TỪ SỐ 0 ĐẾN CHUYÊN GIA](#sec-1)
  - [1.1 Sự Tiến Hóa của Trí Tuệ Nhân Tạo và Thị Giác Máy Tính](#sec-1-1)
  - [1.2 Mạng Nơ-ron Tích Chập (CNN) và Cơ Sở Toán Học](#sec-1-2)
    - [1.2.1 Phép Tích Chập 2D và Cơ Chế Bộ Lọc (Kernel/Filter)](#sec-1-2-1)
    - [1.2.2 Vùng Cảm Nhận Thị Giác (Receptive Field)](#sec-1-2-2)
    - [1.2.3 Đệm Viền (Zero Padding) và Bước Nhảy (Stride)](#sec-1-2-3)
    - [1.2.4 Hàm Kích Hoạt Phi Tuyến Tính (Activation Functions)](#sec-1-2-4)
    - [1.2.5 Lớp Chuẩn Hóa Theo Lô (Batch Normalization)](#sec-1-2-5)
    - [1.2.6 Lớp Gộp: Max Pooling vs. Global Average Pooling (GAP)](#sec-1-2-6)
  - [1.3 Kiến Trúc Đột Phá ResNet50 và Khối Thặng Dư (Residual Block)](#sec-1-3)
    - [1.3.1 Vấn Đề Suy Thoái Mô Hình (Degradation Problem)](#sec-1-3-1)
    - [1.3.2 Chứng Minh Toán Học: Cơ Chế Skip Connection (+1 Gradient)](#sec-1-3-2)
    - [1.3.3 Thiết Kế Khối Bottleneck 3 Tầng](#sec-1-3-3)
  - [1.4 Phương Pháp Học Chuyển Giao (Transfer Learning)](#sec-1-4)
  - [1.5 Không Gian Vector Đặc Trưng và Hình Học Độ Đo Tương Đồng](#sec-1-5)
    - [1.5.1 Vector Đặc Trưng (Feature Embeddings)](#sec-1-5-1)
    - [1.5.2 Phép Chuẩn Hóa L2-Norm (Đưa về Siêu Cầu Đơn Vị)](#sec-1-5-2)
    - [1.5.3 Độ Tương Đồng Cosine (Cosine Similarity)](#sec-1-5-3)
    - [1.5.4 Tại Sao Cosine Similarity Vượt Trội Hơn Khoảng Cách Euclid?](#sec-1-5-4)
  - [1.6 Lý Thuyết Hệ Thống Gợi Ý (Recommender Systems) và Phối Đồ Bù Trừ](#sec-1-6)
    - [1.6.1 Phân Biệt Các Trường Phái Gợi Ý](#sec-1-6-1)
    - [1.6.2 Khác Biệt Cốt Lõi: Gợi Ý Tương Tự vs Gợi Ý Bù Trừ](#sec-1-6-2)
    - [1.6.3 Hàm Chấm Điểm Phối Đồ Đa Mục Tiêu](#sec-1-6-3)
  - [1.7 Các Độ Đo Đánh Giá Hệ Thống Gợi Ý (Evaluation Metrics)](#sec-1-7)
  - [1.8 Mở Rộng Quy Mô Công Nghiệp (Production Scalability)](#sec-1-8)
- [PHẦN 2: BÀI TOÁN THỜI TRANG VÀ KHÁM PHÁ DỮ LIỆU (EDA)](#sec-2)
  - [2.1 Phát biểu Bài toán Phối đồ Thời trang Thông minh](#sec-2-1)
  - [2.2 Nguồn gốc Tập Dữ liệu Thật Kaggle (Fashion Product Images Dataset)](#sec-2-2)
  - [2.3 Phân tích Khám phá Dữ liệu (EDA) và Tiền xử lý Làm sạch](#sec-2-3)
- [PHẦN 3: KIẾN TRÚC MÔ HÌNH VÀ CÁC MÔ-ĐUN LÕI TRONG THƯ MỤC MODEL/](#sec-3)
  - [3.1 Mô-đun Tiền xử lý Dữ liệu (Model/preprocess.py)](#sec-3-1)
  - [3.2 Mô-đun Huấn luyện & Trích xuất Đặc trưng (Model/train_and_val.py)](#sec-3-2)
  - [3.3 Mô-đun Kiểm thử & Đánh giá Toàn diện (Model/test_model.py)](#sec-3-3)
  - [3.4 Động cơ Gợi ý Phối đồ Bù trừ (Complementary Category Pairing Engine)](#sec-3-4)
- [PHẦN 4: KẾT QUẢ THỰC NGHIỆM VÀ ỨNG DỤNG THỰC TẾ](#sec-4)
  - [4.1 Đánh giá Định lượng: Tốc độ và Bộ nhớ](#sec-4-1)
  - [4.2 Đánh giá Định tính: Phân tích Ca Phối đồ Thực tế và Kết quả Test](#sec-4-2)
  - [4.3 Ứng dụng Giao diện Web Thời gian thực (Thư mục app_web/)](#sec-4-3)
  - [4.4 Kịch bản Thuyết trình 10 Phút Trước Hội đồng AI SIC](#sec-4-4)
  - [4.5 Bộ 30 Câu hỏi Phản biện và Đáp án Điểm 10](#sec-4-5)
- [PHẦN 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN](#sec-5)

---

# PHẦN 1: TỔNG QUAN KIẾN THỨC NỀN TẢNG — TỪ SỐ 0 ĐẾN CHUYÊN GIA
"""

    report_p2 = r"""
---

# PHẦN 2: BÀI TOÁN THỜI TRANG VÀ KHÁM PHÁ DỮ LIỆU (EDA)

### 2.1 Phát biểu Bài toán Phối đồ Thời trang Thông minh
Trong đời sống hiện đại, người tiêu dùng thường mất nhiều thời gian mỗi sáng trước câu hỏi "Hôm nay mặc gì?". Các hệ thống thương mại điện tử hiện nay đa phần chỉ giải quyết bài toán gợi ý món đồ tương tự (Item-to-Item Similarity).  
Dự án này tập trung giải quyết bài toán **Gợi ý Bù trừ Bổ trợ (Complementary Outfit Recommendation)**: Người dùng cung cấp hình ảnh 1 món đồ (ví dụ: Áo), AI phân tích và tự động phối trọn bộ một set đồ hoàn chỉnh gồm Quần/Váy + Giày dép + Phụ kiện đồng điệu về phong cách, màu sắc và hoàn cảnh sử dụng.

---

### 2.2 Nguồn gốc Tập Dữ liệu Thật Kaggle (Fashion Product Images Dataset)
* **Nguồn gốc học thuật**: Dữ liệu thu thập từ sàn thương mại điện tử thời trang **Myntra** (nền tảng bán lẻ lớn nhất Ấn Độ, trực thuộc tập đoàn Flipkart/Walmart).
* **Đóng gói công khai**: Do nhà nghiên cứu **Param Aggarwal** làm sạch và công bố trên **Kaggle** (*Fashion Product Images Small*).
* **Quy mô trên hệ thống**: Gồm 1,200 sản phẩm thật với hình ảnh độ phân giải cao tại `dataset/images/` và bảng thông tin thuộc tính tại `dataset/styles.csv`.

---

### 2.3 Phân tích Khám phá Dữ liệu (EDA) và Tiền xử lý Làm sạch
* **Phân bố danh mục (subCategory)**: Topwear chiếm 44.7% (537 items), Shoes chiếm 25.1% (301 items), Bottomwear chiếm 12.8% (153 items), Accessories chiếm 17.4% (209 items).
* **Phân bố giới tính (gender)**: Men chiếm 55.4%, Women chiếm 41.2%, Unisex chiếm 3.4%.
* **Kiểm tra dữ liệu khuyết thiếu**: 100% các dòng metadata đều có file ảnh thực tế nguyên vẹn trên đĩa, loại bỏ hoàn toàn các file hỏng hoặc lỗi dung lượng 0 byte.

---

# PHẦN 3: KIẾN TRÚC MÔ HÌNH VÀ CÁC MÔ-ĐUN LÕI TRONG THƯ MỤC MODEL/

Toàn bộ lõi kỹ thuật của mô hình được thiết kế độc lập, module hóa thành 3 tệp Python chuẩn mực trong thư mục `Model/`:

### 3.1 Mô-đun Tiền xử lý Dữ liệu (`Model/preprocess.py`)
* Đọc bảng dữ liệu thuộc tính từ `dataset/styles.csv`, làm sạch giá trị thiếu (missing values) và kiểm tra tính toàn vẹn của 1,200 tệp ảnh thực tế.
* Thiết lập Transform chuẩn ImageNet: Resize về $224 \times 224 \rightarrow$ Chuyển Tensor $\rightarrow$ Chuẩn hóa phân phối Z-Score với mean và std của ImageNet.
* Phân chia tập dữ liệu thành Train (90%) và Validation (10%).

---

### 3.2 Mô-đun Huấn luyện & Trích xuất Đặc trưng (`Model/train_and_val.py`)
* **Kiến trúc Deep Learning chính (ResNet50 Transfer Learning)**:
  * Đóng băng trọng số các tầng tích chập của mạng ResNet50 tiền huấn luyện ImageNet.
  * Cắt bỏ tầng Fully Connected (FC) 1,000 lớp, giữ lại tầng Global Average Pooling (GAP) để trích xuất vector ngữ nghĩa **2048 chiều**.
* **Kiến trúc Baseline đối chứng (Color Histogram HSV 3D)**:
  * Trích xuất biểu đồ phân bố màu sắc không gian HSV với 8 bins trên mỗi kênh ($8 \times 8 \times 8 = 512$ chiều).
* **Quy trình xử lý theo lô (Batch Processing)**:
  * Sử dụng PyTorch DataLoader (`batch_size = 32`) trích xuất theo lô giúp tăng tốc gấp 5 lần trên CPU.
  * Chuẩn hóa $L_2$-Norm cho từng vector đặc trưng ($\|v\|_2 = 1.0$).
* **Kiểm định chất lượng ma trận (Validation & Sanity Checks)**:
  * Kiểm tra và loại trừ triệt để giá trị NaN, Inf.
  * Kiểm tra độ dài chuẩn L2 của toàn bộ 1,200 vector luôn xấp xỉ $1.0000$.
  * Đóng gói ma trận đặc trưng vào tệp nhị phân `models/embeddings.pkl` ($1200 \times 2048$).

---

### 3.3 Mô-đun Kiểm thử & Đánh giá Toàn diện (`Model/test_model.py`)
* Chạy độc lập qua lệnh `python Model/test_model.py` phục vụ việc báo cáo kỹ thuật trước hội đồng.
* **Test 1 - Đánh giá định lượng**: Đo lường độ trễ truy vấn Cosine Similarity trên toàn kho (< 1ms/sản phẩm), đo bộ nhớ ma trận.
* **Test 2 - Đánh giá định tính**: Thực thi 3 Test Cases thực tế (Nam công sở, Nữ dạo phố, Thể thao) và tự động xuất ảnh trực quan set đồ vào thư mục `Model/test_results/`.
* **Test 3 - Kiểm thử độ bền (Robustness)**: Kiểm tra hệ thống trước các truy vấn lạ và vector nhiễu ngẫu nhiên, chứng minh tính ổn định tuyệt đối.

---

### 3.4 Động cơ Gợi ý Phối đồ Bù trừ (Complementary Category Pairing Engine)
* **Bộ lọc danh mục**: $\text{Topwear} \iff \text{Bottomwear} + \text{Shoes} + \text{Accessories}$.
* **Hàm mục tiêu tính điểm tương thích**:
  $$\text{Score}(q, c) = 0.80 \cdot \text{Cosine}(v_q, v_c) + 0.12 \cdot \mathbb{I}_{(\text{usage}_q = \text{usage}_c)} + 0.20$$
  Giới hạn điểm trong thang đo $[50\%, 99\%]$.

---

# PHẦN 4: KẾT QUẢ THỰC NGHIỆM VÀ ỨNG DỤNG THỰC TẾ

### 4.1 Đánh giá Định lượng: Tốc độ và Bộ nhớ
* **Số chiều vector**: 2048 chiều (ResNet50) vs 512 chiều (Baseline).
* **Độ trễ truy vấn Cosine**: Chỉ **0.635 ms/sản phẩm** trên CPU (đạt tốc độ ~1574 truy vấn/giây).
* **Bộ nhớ**: ~9.8 MB cho ma trận 1,200 sản phẩm.

---

### 4.2 Đánh giá Định tính: Phân tích Ca Phối đồ Thực tế và Kết quả Test
* **Kết quả Test Case 1 (Nam công sở)**: Áo sơ mi nam Mark Taylor Men Plain Blue Shirts $\rightarrow$ AI gợi ý Quần tây Peter England Men Stripes Dark Brown Trouser (84.4%), Giày tây/Sneaker, Đồng hồ Casio G-Shock. Ảnh minh họa được lưu tại `Model/test_results/test_case_men_formal.png`.
* **Kết quả Test Case 2 (Nữ dạo phố)**: Áo thun Flying Machine Women Printed Green T-shirt $\rightarrow$ AI gợi ý Quần short bò Jealous 21 (78.5%), Giày flats Catwalk, Balo Wildcraft (59.9%). Ảnh minh họa được lưu tại `Model/test_results/test_case_women_casual.png`.
* **Đối chứng với Baseline**: Baseline chỉ tìm các món cùng màu sắc một cách đơn điệu, trong khi ResNet50 hiểu được phong cách và phối màu tương phản hài hòa.

---

### 4.3 Ứng dụng Giao diện Web Thời gian thực (Thư mục `app_web/`)
* **Kiến trúc Full-Stack**:
  * Frontend: `templates/index.html` và `static/css/style.css`, `static/js/main.js`.
  * Backend API: `backend.py` (Flask REST API) và `model_caller.py` (Inference Engine).
  * Khởi chạy 1-Click: `start_web.bat` hoặc `python app_web/run_web.py`.
  * Thay thế: Hỗ trợ cả Streamlit qua `streamlit_app.py`.
* Cho phép chọn nhanh các sản phẩm thật đại diện hoặc upload ảnh cá nhân từ máy tính.
* *(Lưu ý báo cáo: Phần giao diện Web này được thực hiện với sự hỗ trợ của AI nhằm phục vụ demo trực quan trước hội đồng).*

---

### 4.4 Kịch bản Thuyết trình 10 Phút Trước Hội đồng AI SIC
Kịch bản chi tiết 10 phút gồm 5 chặng: Mở đầu bài toán $\rightarrow$ EDA $\rightarrow$ Kiến trúc ResNet50 & 3 File Model lõi $\rightarrow$ Live Demo Web $\rightarrow$ Kết luận và Q&A. Đầy đủ trong tệp `Thuyet_Trinh/Tai_Lieu_Thuyet_Trinh_Va_Cau_Hoi_Phan_Bien.pdf`.

---

### 4.5 Bộ 30 Câu hỏi Phản biện và Đáp án Điểm 10
Đầy đủ 30 câu hỏi phản biện chuyên sâu (Toán học, Kiến trúc CNN, Cosine Similarity, Bẫy số chiều, Mở rộng FAISS quy mô lớn) kèm câu trả lời mẫu ghi điểm tuyệt đối trong thư mục `Thuyet_Trinh/`.

---

# PHẦN 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
* **Thành tựu**: Hoàn thành trọn vẹn 3 Stage chuẩn Samsung Innovation Campus (AI SIC).
* **Đóng góp**: Xây dựng thành công hệ thống gợi ý phối đồ thời trang thông minh, độ trễ thấp (< 30ms), giao diện trực quan, kiến trúc module hóa sạch sẽ.
* **Hướng phát triển**: Tích hợp API thời tiết thời gian thực, bóc tách trang phục ngoại cảnh bằng YOLOv8-Seg và thử đồ ảo (Virtual Try-on bằng Generative Diffusion).
"""

    full_report_md = report_p1 + FOUNDATIONAL_KNOWLEDGE + report_p2
    report_md_path = os.path.join(DOCS_DIR, "report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(full_report_md)
    print(f"✅ Đã ghi tệp: {report_md_path}")

    report_html_path = os.path.join(DOCS_DIR, "report.html")
    html_content = make_html_with_working_links(full_report_md, cover_type="report")
    with open(report_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✅ Đã ghi tệp: {report_html_path}")

    report_pdf_path = os.path.join(DOCS_DIR, "Report AI SIC.pdf")
    convert_html_to_pdf(report_html_path, report_pdf_path)

# ==============================================================================
# 2. BIÊN SOẠN CẨM NANG README.MD & README AI SIC.PDF
# ==============================================================================
def generate_readme():
    print("\n[2/2] Đang tạo README.md và README AI SIC.pdf...")

    readme_p1 = r"""# README

---

## MỤC LỤC TỔNG QUAN

- [PHẦN 1: TỔNG QUAN KIẾN THỨC VÀ KIẾN TRÚC MÔ HÌNH HỌC SÂU](#sec-1)
  - [1.1 Sự Tiến Hóa của Trí Tuệ Nhân Tạo và Thị Giác Máy Tính](#sec-1-1)
  - [1.2 Mạng Nơ-ron Tích Chập (CNN) và Cơ Sở Toán Học](#sec-1-2)
    - [1.2.1 Phép Tích Chập 2D và Cơ Chế Bộ Lọc (Kernel/Filter)](#sec-1-2-1)
    - [1.2.2 Vùng Cảm Nhận Thị Giác (Receptive Field)](#sec-1-2-2)
    - [1.2.3 Đệm Viền (Zero Padding) và Bước Nhảy (Stride)](#sec-1-2-3)
    - [1.2.4 Hàm Kích Hoạt Phi Tuyến Tính (Activation Functions)](#sec-1-2-4)
    - [1.2.5 Lớp Chuẩn Hóa Theo Lô (Batch Normalization)](#sec-1-2-5)
    - [1.2.6 Lớp Gộp: Max Pooling vs. Global Average Pooling (GAP)](#sec-1-2-6)
  - [1.3 Kiến Trúc Đột Phá ResNet50 và Khối Thặng Dư (Residual Block)](#sec-1-3)
    - [1.3.1 Vấn Đề Suy Thoái Mô Hình (Degradation Problem)](#sec-1-3-1)
    - [1.3.2 Chứng Minh Toán Học: Cơ Chế Skip Connection (+1 Gradient)](#sec-1-3-2)
    - [1.3.3 Thiết Kế Khối Bottleneck 3 Tầng](#sec-1-3-3)
  - [1.4 Phương Pháp Học Chuyển Giao (Transfer Learning)](#sec-1-4)
  - [1.5 Không Gian Vector Đặc Trưng và Hình Học Độ Đo Tương Đồng](#sec-1-5)
    - [1.5.1 Vector Đặc Trưng (Feature Embeddings)](#sec-1-5-1)
    - [1.5.2 Phép Chuẩn Hóa L2-Norm (Đưa về Siêu Cầu Đơn Vị)](#sec-1-5-2)
    - [1.5.3 Độ Tương Đồng Cosine (Cosine Similarity)](#sec-1-5-3)
    - [1.5.4 Tại Sao Cosine Similarity Vượt Trội Hơn Khoảng Cách Euclid?](#sec-1-5-4)
  - [1.6 Lý Thuyết Hệ Thống Gợi Ý (Recommender Systems) và Phối Đồ Bù Trừ](#sec-1-6)
    - [1.6.1 Phân Biệt Các Trường Phái Gợi Ý](#sec-1-6-1)
    - [1.6.2 Khác Biệt Cốt Lõi: Gợi Ý Tương Tự vs Gợi Ý Bù Trừ](#sec-1-6-2)
    - [1.6.3 Hàm Chấm Điểm Phối Đồ Đa Mục Tiêu](#sec-1-6-3)
  - [1.7 Các Độ Đo Đánh Giá Hệ Thống Gợi Ý (Evaluation Metrics)](#sec-1-7)
  - [1.8 Mở Rộng Quy Mô Công Nghiệp (Production Scalability)](#sec-1-8)
- [PHẦN 2: BẢN ĐỒ ĐIỀU HƯỚNG VÀ TRA CỨU TỆP (FILE LOOKUP MAP & NAVIGATION GUIDE)](#sec-2)
  - [2.1 Sơ đồ Cấu trúc Cây Thư mục Dự án Toàn diện](#sec-2-1)
  - [2.2 Bảng Tra cứu Nhanh: "Tìm gì ở đâu và File nào làm gì?"](#sec-2-2)
  - [2.3 Thư mục Model/: Lõi Thuật toán và Mô hình (Học viên Báo cáo)](#sec-2-3)
  - [2.4 Thư mục app_web/: Giao diện và Gọi Mô hình (AI Hỗ trợ Viết)](#sec-2-4)
  - [2.5 Thư mục Thuyet_Trinh/: Slides PPTX, PDF và Kịch bản Q&A](#sec-2-5)
  - [2.6 Thư mục Report_and_Readme/: Tài liệu và Script PDF (AI Hỗ trợ Viết)](#sec-2-6)
- [PHẦN 3: QUÁ TRÌNH THỰC HIỆN DỰ ÁN (END-TO-END WORKFLOW 6 BƯỚC)](#sec-3)
  - [3.1 Giai đoạn 1: Khảo sát Bài toán và Thiết kế 48 Giờ](#sec-3-1)
  - [3.2 Giai đoạn 2: Tải và Chuẩn hóa Dữ liệu Thật Kaggle](#sec-3-2)
  - [3.3 Giai đoạn 3: Huấn luyện và Trích xuất Đặc trưng](#sec-3-3)
  - [3.4 Giai đoạn 4: Thiết kế Động cơ Phối đồ Bù trừ](#sec-3-4)
  - [3.5 Giai đoạn 5: Đóng gói và Xây dựng Giao diện Web](#sec-3-5)
  - [3.6 Giai đoạn 6: Đánh giá Thực nghiệm và Biên soạn Tài liệu](#sec-3-6)
- [PHẦN 4: HƯỚNG DẪN CÀI ĐẶT VÀ KHỞI CHẠY HỆ THỐNG](#sec-4)
  - [4.1 Cài đặt Thư viện Phụ thuộc](#sec-4-1)
  - [4.2 Khởi chạy Ứng dụng Web (HTML/CSS/JS & Flask API)](#sec-4-2)
  - [4.3 Khởi chạy Phiên bản Streamlit](#sec-4-3)
  - [4.4 Chạy Bộ Kiểm thử Mô hình (Test Model)](#sec-4-4)
  - [4.5 Chạy Tái huấn luyện & Trích xuất Mô hình](#sec-4-5)
  - [4.6 Mở và Chạy Jupyter Notebook 3 Stage](#sec-4-6)
- [PHẦN 5: KẾT LUẬN VÀ HƯỚNG MỞ RỘNG TƯƠNG LAI](#sec-5)

---

# PHẦN 1: TỔNG QUAN KIẾN THỨC VÀ KIẾN TRÚC MÔ HÌNH HỌC SÂU
"""

    readme_p2 = r"""
---

# PHẦN 2: BẢN ĐỒ ĐIỀU HƯỚNG VÀ TRA CỨU TỆP (FILE LOOKUP MAP & NAVIGATION GUIDE)

### 2.1 Sơ đồ Cấu trúc Cây Thư mục Dự án Toàn diện

Toàn bộ dự án trong thư mục `c:\Users\doanm\Desktop\AI SIC` được quy hoạch gọn gàng, tinh giản tuyệt đối. Thư mục gốc chỉ giữ duy nhất file **README** và file thư viện phụ thuộc:

```
c:\Users\doanm\Desktop\AI SIC\
│
├── README.md                     [📄 HƯỚNG DẪN KỸ THUẬT & BẢN ĐỒ TRA CỨU TỆP (ĐỌC TRÊN GITHUB)]
├── README AI SIC.pdf             [📄 CẨM NANG KỸ THUẬT PDF (BÌA: README, CÔNG THỨC MATHML SẮC NÉT)]
├── requirements.txt              [📦 DANH SÁCH THƯ VIỆN PYTHON PHỤ THUỘC TOÀN HỆ THỐNG]
│
├── Thuyet_Trinh/                 [🎤 BỘ TÀI LIỆU VÀ SLIDES THUYẾT TRÌNH BẢO VỆ CAPSTONE]
│   ├── Slides_Thuyet_Trinh.pptx  [🌟 BỘ SLIDES POWERPOINT 16:9 CHUẨN ĐỒ HỌA DARK MODE]
│   ├── Slides_Thuyet_Trinh.pdf   [Bản PDF Slide 16:9 Landscape chuẩn in ấn, không tràn chữ]
│   ├── slides.html               [Bộ 12 Slide 16:9 trình chiếu trực tiếp trên Trình duyệt F11]
│   ├── Tai_Lieu_Thuyet_Trinh_Va_Cau_Hoi_Phan_Bien.pdf [Cẩm nang thuyết trình 10p + 30 câu hỏi điểm 10 (18 trang)]
│   ├── Tai_Lieu_Thuyet_Trinh_Va_Cau_Hoi_Phan_Bien.md  [Mã nguồn Markdown cẩm nang thuyết trình]
│   └── create_presentation_pptx.py [Script tự động xuất Slides PowerPoint]
│
├── Model/                        [📁 THƯ MỤC LÕI: HỌC VIÊN LÀM CHỦ ĐỂ BÁO CÁO VỚI HỘI ĐỒNG]
│   ├── preprocess.py             [Mô-đun 1: Đọc CSV, lọc ảnh lỗi, Transform ImageNet, Train/Val Split]
│   ├── train_and_val.py          [Mô-đun 2: Trích xuất ResNet50 (2048D) & Baseline (512D), Validate]
│   ├── test_model.py             [Mô-đun 3: Kiểm thử định lượng, Test Cases phối đồ, Robustness test]
│   ├── fashion_recommender.ipynb [Notebook thực nghiệm 3 Stage chuẩn Samsung Innovation Campus]
│   └── test_results/             [Thư mục ảnh kết quả trực quan hóa kiểm thử test_case_*.png]
│
├── app_web/                      [🌐 THƯ MỤC WEB & GIAO DIỆN ĐẦY ĐỦ: BÁO CÁO CÓ AI HỖ TRỢ VIẾT]
│   ├── templates/index.html      [🌟 Giao diện Web HTML5 cao cấp, thiết kế hiện đại, responsive]
│   ├── static/css/style.css      [Định kiểu giao diện Glassmorphism, animations, cards]
│   ├── static/js/main.js         [Xử lý Drag & Drop, AJAX gọi API, render gợi ý thời gian thực]
│   ├── backend.py                [Máy chủ Flask REST API xử lý yêu cầu và điều phối mô hình]
│   ├── model_caller.py           [Tầng suy luận: Nạp embeddings, gọi model và phối đồ bù trừ]
│   ├── run_web.py                [Script khởi chạy Web Server và tự mở trình duyệt 1 lệnh]
│   ├── start_web.bat             [File nhấp đúp chuột để chạy ngay trên Windows]
│   ├── streamlit_app.py          [Giao diện Streamlit thay thế]
│   ├── app.py                    [Cổng khởi động đa năng: Chạy Web HTML hoặc Streamlit]
│   └── README_WEB.md             [Hướng dẫn kiến trúc giao diện Web chi tiết]
│
├── Report_and_Readme/            [📚 THƯ MỤC TÀI LIỆU & CODE DỊCH PDF: UP GIT, BÁO CÁO AI HỖ TRỢ]
│   ├── Report AI SIC.pdf         [Bản Báo cáo Tốt nghiệp chính thức (Bìa: BÁO CÁO - Đoàn Mạnh Dũng)]
│   ├── README AI SIC.pdf         [Bản sao lưu Cẩm nang kỹ thuật có mục lục liên kết nhảy trang]
│   ├── report.md & report.html   [Mã nguồn Markdown và HTML in ấn của Báo cáo chính thức]
│   ├── README.html               [Mã nguồn HTML in ấn của Cẩm nang kỹ thuật]
│   ├── Bao_Cao_Va_On_Tap_Thuyet_Trinh.md [Cẩm nang sinh tồn: Từ điển, Kịch bản 10p, 25 Q&A]
│   └── build_all_documents_and_pdfs.py   [Script tự động biên soạn và kết xuất tài liệu sang PDF]
│
├── dataset/                      [📁 DỮ LIỆU ĐẦU VÀO (INPUT): 1,200 SẢN PHẨM THẬT KAGGLE]
│   ├── styles.csv                [Bảng nhãn thuộc tính sản phẩm từ sàn Myntra]
│   ├── images/                   [1,200 tệp ảnh JPG sản phẩm thật độ phân giải cao]
│   └── fashion_raw.parquet       [Tệp nhị phân gốc 136 MB tải từ Kaggle mirror]
│
└── models/                       [📁 ĐẦU RA (OUTPUT): MA TRẬN VECTOR ĐẶC TRƯNG ĐÃ TRÍCH XUẤT]
    ├── embeddings.pkl            [Ma trận ResNet50 1200 x 2048 đã chuẩn hóa L2-Norm]
    ├── baseline_embeddings.pkl   [Ma trận Baseline màu sắc HSV 1200 x 512]
    └── metadata.csv              [Bảng metadata sản phẩm đồng bộ phục vụ truy vấn]
```

---

### 2.2 Bảng Tra cứu Nhanh: "Tìm gì ở đâu và File nào làm gì?"

| Nhu cầu của bạn hoặc Câu hỏi của Giám khảo | Mở File nào? | Vị trí thư mục | Vai trò và Cách giải thích trước Hội đồng |
| :--- | :--- | :--- | :--- |
| **"Slide PowerPoint (.pptx) ở đâu?"** | `Slides_Thuyet_Trinh.pptx` | `Thuyet_Trinh/` | **Slide PowerPoint 16:9 Widescreen**: Thiết kế hiện đại, đầy đủ 12 slide với thẻ trực quan, sẵn sàng chiếu và chỉnh sửa trên Microsoft PowerPoint. |
| **"Slide PDF 16:9 ở đâu?"** | `Slides_Thuyet_Trinh.pdf` | `Thuyet_Trinh/` | **Slide PDF chuẩn in ấn**: 12 trang 16:9 không tràn chữ, trình chiếu Fullscreen. |
| **"Kịch bản nói từng slide & 30 câu hỏi phản biện ở đâu?"** | `Tai_Lieu_Thuyet_Trinh_Va_Cau_Hoi_Phan_Bien.pdf` | `Thuyet_Trinh/` | **Cẩm nang bảo vệ điểm 10**: Lời thoại chi tiết từng câu chữ trong 10 phút và đáp án 30 câu phản biện. |
| **"Code tiền xử lý và chuẩn hóa ảnh ở đâu?"** | `preprocess.py` | `Model/` | **Code lõi tự làm**: Làm sạch dữ liệu, kiểm tra ảnh, Resize $224 \times 224$, Normalize chuẩn ImageNet. |
| **"Code huấn luyện & trích xuất đặc trưng ở đâu?"** | `train_and_val.py` | `Model/` | **Code lõi tự làm**: Dùng PyTorch ResNet50 (2048D) và Baseline HSV (512D), xử lý theo lô Batch size 32, chuẩn hóa $L_2$. |
| **"Code kiểm thử (Test) mô hình ở đâu?"** | `test_model.py` | `Model/` | **Code lõi tự làm**: Benchmark độ trễ (< 1ms), chạy 3 Test Cases phối đồ thực tế và kiểm thử vector nhiễu. |
| **"Tệp Notebook 3 Stage nộp cho SIC ở đâu?"** | `fashion_recommender.ipynb` | `Model/` | **Thực nghiệm chuẩn**: Notebook 33 cells theo đúng chuẩn 3 Stage của Samsung Innovation Campus. |
| **"Giao diện Web HTML/CSS/JS ở đâu?"** | `templates/index.html` & `static/` | `app_web/` | **Giao diện Web chuyên nghiệp**: HTML5, CSS Glassmorphism, JavaScript xử lý upload và hiển thị kết quả. |
| **"Máy chủ Backend API ở đâu?"** | `backend.py` | `app_web/` | **Máy chủ Flask REST API**: Tiếp nhận yêu cầu gợi ý và kết nối với model. |
| **"Cách mở nhanh Web trên máy tính?"** | `start_web.bat` hoặc `run_web.py` | `app_web/` | **1-Click Launch**: Nhấp đúp chuột là tự động khởi động server và mở trình duyệt tại `http://127.0.0.1:5000`. |
| **"Báo cáo tốt nghiệp chính thức ở đâu?"** | `Report AI SIC.pdf` | `Report_and_Readme/` | Báo cáo chính thức: Bìa chỉ có **BÁO CÁO - Đoàn Mạnh Dũng**, công thức toán MathML sắc nét, mục lục nhảy trang 100%. |

---

### 2.3 Thư mục `Model/`: Lõi Thuật toán và Mô hình (Học viên Báo cáo)
Đây là thư mục quan trọng nhất mà học viên cần **học hiểu sâu từng dòng code** để tự tin trình bày và bảo vệ trước hội đồng:
1. **`preprocess.py`**: Định nghĩa lớp `FashionDataset`, thiết lập bộ lọc làm sạch dữ liệu, kiểm tra tệp ảnh và pipeline chuyển đổi hình học Tensor.
2. **`train_and_val.py`**: Định nghĩa kiến trúc `ResNet50FeatureExtractor`, đóng băng trọng số, khai thác tầng GAP 2048 chiều, kiểm định độ chuẩn hóa $L_2$-Norm ($\|v\|_2 = 1.0$) và loại bỏ lỗi NaN/Inf.
3. **`test_model.py`**: Bộ kiểm thử tự động toàn diện. Chạy lệnh `python Model/test_model.py` sẽ thực hiện bài test định lượng (đo độ trễ < 1ms), bài test định tính (gợi ý set đồ cho nam và nữ) và tự động chụp ảnh kết quả lưu vào `Model/test_results/`.
4. **`fashion_recommender.ipynb`**: File notebook thực nghiệm 3 Stage dùng để nộp chấm điểm phần thực hành.

---

### 2.4 Thư mục `app_web/`: Giao diện và Gọi Mô hình (AI Hỗ trợ Viết)
Thư mục này phục vụ mục đích chuyển hóa mô hình thành sản phẩm ứng dụng thực tế:
* Cấu trúc rõ ràng: Tách bạch rõ rệt giữa Frontend (`templates/index.html`, `static/css`, `static/js`), Backend API (`backend.py`), Inference Engine (`model_caller.py`) và Runner (`run_web.py`, `start_web.bat`).
* Học viên có thể trả lời hội đồng: *"Để minh họa trực quan kết quả nghiên cứu, em đã ứng dụng các công cụ AI hỗ trợ lập trình nhanh giao diện Web tương tác và máy chủ Flask API trong thư mục app_web/."*

---

### 2.5 Thư mục `Thuyet_Trinh/`: Slides PPTX, PDF và Kịch bản Q&A
Chứa trọn bộ tài liệu bảo vệ:
* `Slides_Thuyet_Trinh.pptx`: File PowerPoint 16:9 Widescreen hiện đại, bố cục thẻ không tràn chữ.
* `Slides_Thuyet_Trinh.pdf` & `slides.html`: Slide PDF và Web phục vụ trình chiếu linh hoạt.
* `Tai_Lieu_Thuyet_Trinh_Va_Cau_Hoi_Phan_Bien.pdf`: Kịch bản nói chi tiết 10 phút và 30 câu hỏi phản biện điểm 10.

---

### 2.6 Thư mục `Report_and_Readme/`: Tài liệu và Script PDF (AI Hỗ trợ Viết)
Chứa toàn bộ các phiên bản báo cáo, tài liệu ôn thi, các file trung gian HTML và script dịch tự động sang PDF với công thức toán MathML chuẩn xác.

---

# PHẦN 3: QUÁ TRÌNH THỰC HIỆN DỰ ÁN (END-TO-END WORKFLOW 6 BƯỚC)

```
[BƯỚC 1: KHẢO SÁT & THIẾT KẾ 48H]
       │  * Xác định bài toán Phối đồ bù trừ (Complementary Outfit Recommendation)
       │  * Chọn giải pháp: Pretrained ResNet50 + L2-Norm + Cosine Similarity
       ▼
[BƯỚC 2: TẢI & CHUẨN HÓA DỮ LIỆU THẬT KAGGLE]
       │  * Chạy download_real_kaggle_dataset.py tải 136MB Parquet từ Kaggle mirror
       │  * Giải nén 1,200 ảnh sản phẩm thật và metadata styles.csv vào dataset/
       ▼
[BƯỚC 3: XÂY DỰNG 3 FILE LÕI MODEL/ (TIỀN XỬ LÝ - TRAIN & VAL - TEST)]
       │  * Model/preprocess.py: Làm sạch dữ liệu, ImageNet Transform
       │  * Model/train_and_val.py: Trích xuất ResNet50 (2048D) & Baseline (512D) theo lô
       │  * Model/test_model.py: Kiểm thử định lượng (< 1ms), định tính và xuất ảnh
       ▼
[BƯỚC 4: THIẾT KẾ ĐỘNG CƠ PHỐI ĐỒ BÙ TRỪ]
       │  * Luật bù trừ: Topwear <-> Bottomwear + Shoes + Accessories
       │  * Hàm mục tiêu kết hợp Cosine Similarity và Style Bonus
       ▼
[BƯỚC 5: ĐÓNG GÓI WEB FULL-STACK (HTML/CSS/JS + FLASK API)]
       │  * Xây dựng giao diện templates/index.html & static/ (AI hỗ trợ)
       │  * Xây dựng app_web/backend.py & run_web.py / start_web.bat
       │  * Hỗ trợ chọn nhanh kho mẫu hoặc upload ảnh cá nhân
       ▼
[BƯỚC 6: ĐÁNH GIÁ THỰC NGHIỆM & BIÊN SOẠN TÀI LIỆU BẢO VỆ]
          * Tạo fashion_recommender.ipynb bám sát 3 Stage AI SIC
          * Biên soạn Slides_Thuyet_Trinh.pptx, Report AI SIC.pdf và cẩm nang 30 Q&A
```

---

# PHẦN 4: HƯỚNG DẪN CÀI ĐẶT VÀ KHỞI CHẠY HỆ THỐNG

### 4.1 Cài đặt Thư viện Phụ thuộc
Mở terminal tại thư mục dự án và cài đặt:
```bash
pip install -r requirements.txt
```

### 4.2 Khởi chạy Ứng dụng Web (HTML/CSS/JS & Flask API)
Có 2 cách đơn giản:
* **Cách 1 (Nhanh nhất trên Windows)**: Nhấp đúp chuột vào file `app_web/start_web.bat`.
* **Cách 2 (Dùng lệnh terminal)**:
```bash
python app_web/run_web.py
```
* Trình duyệt sẽ tự động mở giao diện tại `http://127.0.0.1:5000`.

### 4.3 Khởi chạy Phiên bản Streamlit (Nếu muốn)
```bash
streamlit run app_web/streamlit_app.py
```

### 4.4 Chạy Bộ Kiểm thử Mô hình (Test Model)
Để kiểm tra toàn diện hiệu năng và xuất ảnh minh họa kết quả phối đồ:
```bash
python Model/test_model.py
```
* Kết quả test sẽ in ra terminal và ảnh trực quan sẽ được lưu tại `Model/test_results/`.

### 4.5 Chạy Tái huấn luyện & Trích xuất Mô hình
Nếu bạn bổ sung thêm sản phẩm mới vào `dataset/`:
```bash
python Model/train_and_val.py
```

### 4.6 Mở và Chạy Jupyter Notebook 3 Stage
```bash
jupyter notebook Model/fashion_recommender.ipynb
```

---

# PHẦN 5: KẾT LUẬN VÀ HƯỚNG MỞ RỘNG TƯƠNG LAI

### 5.1 Kết luận Tổng kết
Hệ thống Gợi ý Phối đồ Thời trang Thông minh (Smart Fashion Outfit Recommender System) đã giải quyết trọn vẹn và xuất sắc bài toán đặt ra theo chuẩn 3 Stage của chương trình **Samsung Innovation Campus (AI SIC)**:
1. **Làm chủ dữ liệu**: Làm sạch và tiền xử lý 1,200 sản phẩm thật từ tập dữ liệu quốc tế Kaggle Fashion Product Images (`Model/preprocess.py`).
2. **Làm chủ công nghệ Deep Learning**: Ứng dụng thành công Transfer Learning với ResNet50, trích xuất không gian đặc trưng ngữ nghĩa 2048 chiều, kiểm định chất lượng ma trận và kiểm thử định lượng/định tính toàn diện (`Model/train_and_val.py` và `Model/test_model.py`).
3. **Đóng gói sản phẩm thực tế**: Triển khai ứng dụng Web hoàn thiện với kiến trúc Frontend/Backend tách bạch, phản hồi tức thì dưới 30ms (`app_web/`).

### 5.2 Hướng Mở rộng Tương lai
* **Mở rộng quy mô với FAISS / HNSW**: Chuyển đổi sang tìm kiếm láng giềng gần đúng (ANN) khi kho sản phẩm đạt hàng triệu items.
* **Tích hợp API Thời tiết**: Tự động gợi ý outfit theo điều kiện thời tiết thực tế.
* **Bóc tách trang phục ngoại cảnh**: Sử dụng YOLOv8-Seg để tách nền cho các ảnh chụp người thật ngoài đường phố.
* **Thử đồ ảo (Virtual Try-on)**: Tích hợp Generative Diffusion Models để người dùng nhìn thấy hình ảnh chính mình mặc set đồ được gợi ý.
"""

    full_readme_md = readme_p1 + FOUNDATIONAL_KNOWLEDGE + readme_p2

    # Ghi README.md tại ROOT để hiển thị trực tiếp trên GitHub
    root_readme_md = os.path.join(PROJECT_ROOT, "README.md")
    with open(root_readme_md, "w", encoding="utf-8") as f:
        f.write(full_readme_md)
    print(f"✅ Đã ghi tệp ROOT: {root_readme_md}")

    # Bản sao lưu trong DOCS_DIR
    docs_readme_md = os.path.join(DOCS_DIR, "README.md")
    with open(docs_readme_md, "w", encoding="utf-8") as f:
        f.write(full_readme_md)

    # Ghi README.html trong DOCS_DIR
    docs_readme_html = os.path.join(DOCS_DIR, "README.html")
    html_content = make_html_with_working_links(full_readme_md, cover_type="readme")
    with open(docs_readme_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✅ Đã ghi tệp DOCS: {docs_readme_html}")

    # Xuất README AI SIC.pdf ra cả ROOT và DOCS_DIR
    root_readme_pdf = os.path.join(PROJECT_ROOT, "README AI SIC.pdf")
    convert_html_to_pdf(docs_readme_html, root_readme_pdf)

    docs_readme_pdf = os.path.join(DOCS_DIR, "README AI SIC.pdf")
    shutil.copy2(root_readme_pdf, docs_readme_pdf)
    print(f"✅ Đã sao chép README AI SIC.pdf -> {DOCS_DIR}/")

# ==============================================================================
# DỌN DẸP THƯ MỤC ROOT: CHỈ ĐỂ DUY NHẤT CÁC FILE README
# ==============================================================================
def clean_root_directory():
    print("\n🧹 Đang kiểm tra và dọn dẹp thư mục gốc (Root)...")
    forbidden_at_root = [
        "Report AI SIC.pdf",
        "report.md",
        "report.html",
        "README.html",
        "Bao_Cao_Va_On_Tap_Thuyet_Trinh.md",
        "build_all_documents_and_pdfs.py",
        "create_full_report.py",
        "app.py",
        "fashion_recommender.ipynb",
        "build_notebook.py",
        "train_model.py",
        "prepare_data.py",
        "download_real_kaggle_dataset.py",
        "Slides_Thuyet_Trinh.pptx",
        "Slides_Thuyet_Trinh.pdf",
        "slides.html"
    ]
    for fn in forbidden_at_root:
        fp = os.path.join(PROJECT_ROOT, fn)
        if os.path.exists(fp):
            os.remove(fp)
            print(f"   🗑️ Đã xóa file thừa ở root: {fn}")

    forbidden_dirs_at_root = ["test_katex_dir", "data", "__pycache__"]
    for dn in forbidden_dirs_at_root:
        dp = os.path.join(PROJECT_ROOT, dn)
        if os.path.exists(dp):
            shutil.rmtree(dp, ignore_errors=True)
            print(f"   🗑️ Đã xóa thư mục thừa ở root: {dn}")

# ==============================================================================
# HÀM CHÍNH
# ==============================================================================
def main():
    generate_report()
    generate_readme()
    clean_root_directory()

    print("\n" + "="*70)
    print("🎉 HOÀN TẤT BIÊN SOẠN VÀ ĐỒNG BỘ TOÀN DIỆN!")
    print("📁 Thư mục gốc (Root): Chỉ để duy nhất README.md & README AI SIC.pdf (và requirements.txt).")
    print("📁 Thư mục Report_and_Readme/: Chứa đầy đủ Report AI SIC.pdf, report.md, v.v.")
    print("📁 Thư mục Thuyet_Trinh/: Chứa Slides_Thuyet_Trinh.pptx, Slides_Thuyet_Trinh.pdf, v.v.")
    print("📁 Thư mục Model/: Chứa đầy đủ 3 file code lõi và fashion_recommender.ipynb.")
    print("📁 Thư mục app_web/: Chứa đầy đủ ứng dụng web HTML/CSS/JS + Flask API và Streamlit.")
    print("="*70)

if __name__ == "__main__":
    main()
