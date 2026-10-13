# README

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
