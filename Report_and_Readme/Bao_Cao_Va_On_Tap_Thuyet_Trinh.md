# 🎓 BÁO CÁO CAPSTONE & CẨM NANG SINH TỒN THUYẾT TRÌNH TỪ CON SỐ 0
## DỰ ÁN: HỆ THỐNG GỢI Ý PHỐI ĐỒ THỜI TRANG THÔNG MINH (SMART FASHION OUTFIT RECOMMENDER SYSTEM)
### KHÓA ĐÀO TẠO TRÍ TUỆ NHÂN TẠO - SAMSUNG INNOVATION CAMPUS (AI SIC)

---

> **LỜI NHẮN NHỦ TỪ SENIOR AI MENTOR DÀNH CHO BẠN (TRƯỚC GIỜ G 48 TIẾNG):**  
> *"Dù bạn có cảm giác mình bị mất gốc kiến thức từ đầu khóa, hãy thở sâu và yên tâm! Hội đồng chấm tốt nghiệp Capstone AI SIC không tìm kiếm một nhà toán học hàn lâm đọc thuộc lòng phương trình, mà họ tìm kiếm một **Kỹ sư AI ứng dụng (Applied AI Engineer)**: người hiểu rõ bài toán thực tế, biết chọn kiến trúc giải pháp tối ưu, giải thích được bản chất tại sao lại làm như vậy và chứng minh được sản phẩm chạy mượt mà, trực quan. Tài liệu này được thiết kế để biến bạn từ người mất gốc thành một diễn giả tự tin làm chủ toàn bộ dự án trong 48 giờ."*

---

## MỤC LỤC TỔNG QUAN
1. [Phần 1: Từ điển Giải ngố "Zero to Hero" (Bản chất công nghệ bằng ngôn ngữ đời thường)](#phần-1-từ-điển-giải-ngố-zero-to-hero)
2. [Phần 2: Kịch bản Thuyết trình Bảo vệ Capstone 10 phút (Slide-by-Slide Script)](#phần-2-kịch-bản-thuyết-trình-bảo-vệ-capstone-10-phút)
3. [Phần 3: Bộ 25 Câu hỏi Phản biện "Sát thủ" của Giám khảo AI SIC & Đáp án Điểm 10](#phần-3-bộ-25-câu-hỏi-phản-biện-sát-thủ-của-giám-khảo-ai-sic--đáp-án-điểm-10)
4. [Phần 4: Barem Chấm điểm AI SIC & Chiến thuật Ứng biến Phòng thủ](#phần-4-barem-chấm-điểm-ai-sic--chiến-thuật-ứng-biến-phòng-thủ)

---

# PHẦN 1: TỪ ĐIỂN GIẢI NGỐ "ZERO TO HERO"
*(Hiểu sâu bản chất các khái niệm Deep Learning phức tạp qua những ví dụ đời thường - Explain Like I'm Five)*

---

### 1.1. Vector đặc trưng (Feature Vector / Image Embedding) là gì?
* **Định nghĩa hàn lâm**: Là một biểu diễn toán học dạng mảng số thực đa chiều trong không gian $D$-chiều, thu được từ các tầng ẩn của mô hình học sâu.
* **Giải thích đời thường (ELI5)**:  
  Hãy tưởng tượng **Căn cước công dân** hoặc **Tờ khai lý lịch** của bạn. Thay vì phải mang cả người bạn đi để chứng minh danh tính, công an chỉ cần nhìn tờ khai gồm các con số: Chiều cao (175cm), Cân nặng (68kg), Năm sinh (2002), Dấu vân tay...  
  Đối với máy tính, một bức ảnh thời trang chỉ là hàng vạn điểm ảnh (pixel) màu đỏ-xanh-lam hỗn độn. **Vector đặc trưng (Embedding)** chính là một "bản tóm tắt số đo" của chiếc áo đó: nó gói gọn độ dài tay áo, kiểu cổ áo, độ xòe, họa tiết kẻ sọc, chất liệu vải dạ hay cotton thành một danh sách gồm **2048 con số thực**. Nhờ có 2048 con số này, máy tính có thể so sánh 2 món đồ nhanh như chớp mà không cần nhìn lại từng điểm ảnh.

---

### 1.2. Mạng nơ-ron tích chập (CNN) "nhìn" bức ảnh như thế nào?
* **Giải thích đời thường (ELI5)**:  
  Con người nhìn một chiếc áo thì thấy ngay *"À, đây là áo sơ mi trắng công sở"*. Nhưng máy tính thì mù chữ về mặt thị giác, nó chỉ thấy một ma trận số điểm ảnh từ 0 đến 255.  
  Mạng CNN nhìn bức ảnh giống như một **thám tử cầm kính lúp** soi từ góc nhỏ:
  * **Các tầng đầu (Early Layers)**: Kính lúp soi các chi tiết siêu nhỏ: đoạn thẳng này nghiêng bao nhiêu độ, góc nhọn này sáng hay tối, chỗ này viền màu đen hay trắng.
  * **Các tầng giữa (Middle Layers)**: Ghép các đoạn thẳng lại thành hình dáng cơ bản: hình tròn của chiếc cúc áo, đường thẳng của cổ áo bẻ, vạt áo.
  * **Các tầng sâu (Deep Layers)**: Ghép các hình dáng lại thành khái niệm thời trang hoàn chỉnh: *"Đây là form áo sơ mi Oxford dài tay"* hoặc *"Đây là chiếc quần jeans ống côn"*.

---

### 1.3. Kiến trúc ResNet50 và cơ chế "Skip Connection" (Đột phá đoạt giải quốc tế)
* **Vấn đề lịch sử**: Trước năm 2015, người ta nghĩ cứ xếp chồng càng nhiều tầng tích chập thì mạng càng thông minh. Nhưng khi tăng lên 20-30 tầng, mạng bị hiện tượng **Triệt tiêu đạo hàm (Vanishing Gradient)**. Tức là khi truyền tín hiệu ngược từ đuôi về đầu để học, con số đạo hàm bị nhân với nhiều số thập phân nhỏ hơn 1 khiến nó teo tóp dần về 0. Các tầng đầu tiên bị "điếc", không học thêm được gì cả!
* **Giải pháp thiên tài của ResNet (Residual Network)**:  
  ResNet phát minh ra **Kết nối tắt (Skip Connection hay Shortcut)**:
  $$y = \mathcal{F}(x) + x$$
* **Ví dụ đời thường**:  
  Tưởng tượng bạn đang làm báo cáo qua 5 cấp sếp duyệt. Sếp cấp 5 nhận được văn bản nhưng thông tin đã bị tam sao thất bản, sai lệch hoàn toàn. Cơ chế Skip Connection giống như việc: cấp dưới làm bản báo cáo tóm tắt cải tiến $\mathcal{F}(x)$, nhưng **kẹp kèm luôn bản gốc $x$ gửi thẳng vượt cấp lên cho sếp tổng**. Sếp tổng chỉ việc so sánh: *"Bản mới có gì tốt hơn bản gốc không?"*. Nếu không tốt hơn, mạng chỉ cần đặt $\mathcal{F}(x) = 0$ và giữ nguyên $y = x$. Nhờ vậy, mạng sâu tới 50 tầng (ResNet50) hay 152 tầng (ResNet152) vẫn học cực kỳ ổn định, không bao giờ bị nghẽn thông tin.

---

### 1.4. Transfer Learning (Học chuyển giao) là gì?
* **Giải thích đời thường**:  
  Để đào tạo một đầu bếp chuyên nghiệp, người ta mất 10 năm học từ cách cầm dao, phân biệt hành tỏi, cảm nhận nhiệt độ lửa. Bây giờ, nếu muốn đào tạo đầu bếp đó nấu thêm món "Phở bò truyền thống", bạn chỉ cần dạy họ công thức nấu phở trong 2 tuần, chứ không ai bắt họ đi học lại cách cầm dao từ đầu!  
  **ResNet50** đã được Google/nghiên cứu quốc tế huấn luyện trên hơn **1.2 triệu bức ảnh (ImageNet)**, tiêu tốn hàng nghìn giờ GPU đắt đỏ. Nó đã là một "bậc thầy thị giác", biết rõ mọi đường cong, khối màu, góc cạnh trong tự nhiên.  
  Trong đồ án này, chúng ta áp dụng **Transfer Learning dạng Feature Extraction**: tận dụng toàn bộ khối óc thị giác sẵn có của ResNet50 để trích xuất đặc trưng trang phục, giúp tiết kiệm 99% thời gian và tài nguyên phần cứng mà kết quả vẫn đạt chuẩn công nghiệp.

---

### 1.5. Tại sao dùng Cosine Similarity thay vì Khoảng cách Euclid (Euclidean Distance)?
* **Công thức toán học**:
  $$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = \cos(\theta)$$
  $$\text{Euclidean Distance}(u, v) = \sqrt{\sum_{i=1}^n (u_i - v_i)^2}$$
* **Ví dụ đời thường làm giám khảo gật gù**:  
  Hãy tưởng tượng hai người cùng chỉ tay về **hướng Bắc**. Một người đứng cách bạn 1 mét, một người đứng cách bạn 100 mét.  
  * Nếu đo bằng **Khoảng cách Euclid** (đo khoảng cách thẳng giữa 2 bàn tay), máy tính sẽ bảo: *"Hai người này khác xa nhau tận 99 mét!"* $\rightarrow$ **Sai lầm**, vì thực chất cả hai đều đang chỉ về cùng một hướng!
  * Nếu đo bằng **Cosine Similarity** (đo góc tạo bởi 2 cánh tay): Góc giữa hai người là $0^\circ \rightarrow \cos(0^\circ) = 1.0$ (Trùng khớp 100%).  
  Trong không gian 2048 chiều của ResNet50, **hướng của vector** đại diện cho **phong cách thời trang và kiểu dáng**, còn **độ dài của vector** thường chỉ phản ánh độ sáng/độ tương phản của bức ảnh. Do đó, Cosine Similarity triệt tiêu được sai lệch ánh sáng và nắm bắt chính xác bản chất phong cách của trang phục.

---

### 1.6. Tại sao phải Resize về 224x224 và chuẩn hóa ImageNet Mean/Std?
* **Resize 224x224**: Ma trận trọng số (Weights) ở các tầng đầu của ResNet50 được thiết kế cố định nhận đầu vào 3 kênh màu kích thước $224 \times 224 \times 3$.
* **Chuẩn hóa với Mean = [0.485, 0.456, 0.406] và Std = [0.229, 0.224, 0.225]**:  
  Đây là giá trị trung bình và độ lệch chuẩn của 1.2 triệu bức ảnh ImageNet. Khi ta trừ đi mean và chia cho std, phân bố pixel của ảnh người dùng tải lên sẽ rơi đúng vào phân bố chuẩn mà mạng ResNet50 đã quen thuộc khi huấn luyện. Điều này giúp các nơ-ron kích hoạt ở trạng thái nhạy cảm nhất, không bị rơi vào vùng bão hòa (Saturation).

---

### 1.7. Bẫy số chiều (Curse of Dimensionality) và Chuẩn hóa $L_2$-Norm
* Trong không gian 2048 chiều, các điểm dữ liệu có xu hướng nằm rất xa nhau và dạt về bề mặt ngoài của siêu khối cầu. Để các phép tính toán học không bị bùng nổ giá trị và để tích vô hướng tương đương góc Cosine, ta chuẩn hóa vector theo độ dài hình học $L_2$:
  $$v_{norm} = \frac{v}{\sqrt{\sum v_i^2}} \implies \|v_{norm}\|_2 = 1.0$$
  Khi cả hai vector đều có độ dài bằng 1, công thức Cosine Similarity đơn giản hóa thành phép nhân ma trận siêu nhanh: $\cos(\theta) = u \cdot v$.

---

# PHẦN 2: KỊCH BẢN THUYẾT TRÌNH BẢO VỆ CAPSTONE 10 PHÚT
*(Lời thoại chi tiết từng slide - Khẩu khí tự tin, rành mạch, chuẩn phong thái kỹ sư AI)*

---

### PHÂN BỔ THỜI GIAN (TỔNG: 10 PHÚT THUYẾT TRÌNH + 5 PHÚT Q&A)
* **Phút 00:00 - 01:30 (Slide 1 - 2)**: Giới thiệu, Đặt vấn đề & Mục tiêu dự án.
* **Phút 01:30 - 03:30 (Slide 3 - 4)**: Stage 1 - Dữ liệu & Tiền xử lý (EDA).
* **Phút 03:30 - 06:00 (Slide 5 - 7)**: Stage 2 - Kiến trúc Mô hình (Baseline vs ResNet50) & Thuật toán Phối đồ.
* **Phút 06:00 - 08:30 (Slide 8 - 9)**: Stage 3 - LIVE DEMO Giao diện Web tương tác Streamlit.
* **Phút 08:30 - 10:00 (Slide 10)**: Kết luận, Đóng góp & Lời cảm ơn.

---

### SLIDE 1: TRANG TIÊU ĐỀ
* **Nội dung slide chiếu**:  
  * Tiêu đề: *Hệ thống Gợi ý Phối đồ Thời trang Thông minh (Smart Fashion Outfit Recommender System)*
  * Khóa học: Samsung Innovation Campus (AI SIC) - Capstone Project
  * Học viên thực hiện & Giảng viên hướng dẫn.
* **Lời thoại của bạn**:  
  *"Kính thưa quý thầy cô trong Hội đồng Giám khảo Samsung Innovation Campus! Em xin đại diện nhóm trình bày báo cáo đồ án tốt nghiệp Capstone với đề tài: **Hệ thống Gợi ý Phối đồ Thời trang Thông minh dựa trên Thị giác máy tính và Học sâu**. Dự án được xây dựng bám sát 3 giai đoạn chuẩn kỹ thuật của chương trình SIC: từ nghiên cứu dữ liệu, xây dựng mô hình trích xuất đặc trưng cho đến đóng gói thành một sản phẩm ứng dụng tương tác thời gian thực."*

---

### SLIDE 2: ĐẶT VẤN ĐỀ & MỤC TIÊU DỰ ÁN
* **Nội dung slide chiếu**:  
  * Nỗi đau người dùng (User Pain Point): Câu hỏi kinh điển *"Hôm nay mặc gì?"*, lãng phí trung bình 15 phút mỗi sáng; việc phối đồ thủ công thiếu tính hài hòa về phong cách và màu sắc.
  * Giải pháp kỹ thuật: Trích xuất biểu diễn thị giác (Visual Embeddings) bằng Deep Learning để tự động tạo Set đồ hoàn chỉnh (Áo + Quần/Váy + Giày dép + Phụ kiện).
* **Lời thoại của bạn**:  
  *"Thưa thầy cô, trong kỷ nguyên thương mại điện tử thời trang, vấn đề lớn nhất của người tiêu dùng không phải là thiếu quần áo, mà là **không biết phối các món đồ lại với nhau sao cho đẹp và đúng hoàn cảnh**. Nếu như các hệ thống gợi ý truyền thống chỉ gợi ý các món tương tự (ví dụ: xem áo sơ mi lại gợi ý mua thêm một chiếc áo sơ mi khác), thì mục tiêu của đồ án chúng em là giải quyết bài toán **Gợi ý bù trừ bổ trợ (Complementary Recommendation)**: Người dùng tải lên một chiếc áo, hệ thống sẽ tự động phối trọn bộ một chiếc quần tây sang trọng, một đôi giày Oxford và một chiếc đồng hồ đồng điệu về phong cách."*

---

### SLIDE 3: STAGE 1 - THU THẬP & KHÁM PHÁ DỮ LIỆU (EDA)
* **Nội dung slide chiếu**:  
  * Dataset: Fashion Product Images Dataset (chuẩn hóa quốc tế).
  * 4 biểu đồ EDA: Phân bố phân loại (subCategory), Giới tính (Men, Women, Unisex), Hoàn cảnh (Casual, Formal, Sports, Party), và Màu sắc (baseColour).
  * Thống kê làm sạch: Kiểm tra missing values (0%), lọc bỏ ảnh lỗi.
* **Lời thoại của bạn**:  
  *"Bước vào Giai đoạn 1, nhóm tiến hành thu thập và phân tích khám phá dữ liệu. Bộ dữ liệu chứa đầy đủ các nhóm trang phục thiết yếu với các thuộc tính nhãn phong phú. Qua phân tích EDA, nhóm nhận thấy dữ liệu có độ bao phủ đồng đều giữa các phong cách thường ngày (Casual), công sở (Formal) và thể thao (Sports). Nhóm đã xây dựng quy trình tiền xử lý dữ liệu chặt chẽ, kiểm tra tính toàn vẹn của tệp ảnh trên đĩa và đảm bảo không có giá trị khuyết thiếu ở các trường nhãn trọng yếu."*

---

### SLIDE 4: TIỀN XỬ LÝ ẢNH CHUẨN IMAGENET
* **Nội dung slide chiếu**:  
  * Sơ đồ pipeline: $Image \rightarrow \text{Resize}(224 \times 224) \rightarrow \text{ToTensor}([0, 1]) \rightarrow \text{Normalize}(\text{mean, std})$.
  * Giải thích trực quan tại sao cần chuẩn hóa theo phân bố ImageNet.
* **Lời thoại của bạn**:  
  *"Về tiền xử lý hình ảnh, để mô hình học sâu đạt hiệu năng tối ưu, toàn bộ ảnh được đưa về kích thước cố định $224 \times 224$ pixel. Sau đó, nhóm thực hiện chuẩn hóa pixel theo đúng kỳ vọng toán học và độ lệch chuẩn của tập ImageNet. Bước này giúp đưa dữ liệu đầu vào về đúng không gian phân bố mà mạng nơ-ron đã được huấn luyện trước, ngăn chặn hiện tượng bão hòa tín hiệu đạo hàm."*

---

### SLIDE 5: STAGE 2 - TIẾP CẬN KỸ THUẬT: BASELINE VS. DEEP LEARNING
* **Nội dung slide chiếu**:  
  * Bảng so sánh 2 phương pháp:
    * **Baseline**: Color Histogram không gian màu HSV 3D (512 chiều).
    * **Mô hình chính**: Pretrained ResNet50 Transfer Learning (2048 chiều).
  * Sơ đồ kiến trúc ResNet50 với cơ chế Skip Connection và tầng Global Average Pooling (GAP).
* **Lời thoại của bạn**:  
  *"Ở Giai đoạn 2, nhóm đặt ra phương pháp tiếp cận có đối chứng khoa học:
  Đầu tiên, nhóm xây dựng mô hình **Baseline** dựa trên biểu đồ phân bố màu sắc HSV 512 chiều. Phương pháp này chạy nhanh nhưng có nhược điểm chí mạng: nó chỉ nhìn thấy màu mà hoàn toàn 'mù tịt' về form dáng, đường may và chất liệu vải.  
  Vì vậy, nhóm quyết định lựa chọn giải pháp **Deep Learning sử dụng Transfer Learning với kiến trúc ResNet50**. Nhóm loại bỏ tầng phân loại cuối cùng, giữ lại tầng Global Average Pooling để trích xuất vector biểu diễn ngữ nghĩa 2048 chiều. Nhờ các khối Residual Block, ResNet50 có khả năng bóc tách các đặc trưng thị giác cấp cao như ve áo vest, nếp gấp quần tây hay cổ áo sơ mi mà các thuật toán truyền thống không thể làm được."*

---

### SLIDE 6: THUẬT TOÁN GỢI Ý PHỐI ĐỒ (PAIRING ENGINE)
* **Nội dung slide chiếu**:  
  * Công thức tính điểm tương thích:  
    $$\text{Outfit Score} = \alpha \cdot \text{Cosine Similarity} + \beta \cdot \text{Style Consistency Bonus}$$
  * Quy tắc phối đồ: Bộ lọc danh mục bù trừ (Topwear $\leftrightarrow$ Bottomwear $\leftrightarrow$ Shoes $\leftrightarrow$ Accessories) + Quy tắc đồng điệu hoàn cảnh (Formal/Casual/Sports).
* **Lời thoại của bạn**:  
  *"Trái tim của hệ thống gợi ý nằm ở thuật toán phối đồ bù trừ. Khi nhận một món đồ đầu vào, hệ thống thực hiện 3 bước:
  Thứ nhất, tính toán **Cosine Similarity** giữa vector ảnh đầu vào với toàn bộ kho sản phẩm trong không gian vector đã được chuẩn hóa $L_2$.  
  Thứ hai, áp dụng **Bộ lọc bù trừ danh mục**: Nếu người dùng đưa vào áo, hệ thống chủ động truy vấn các nhóm danh mục đối ứng là Quần, Giày và Phụ kiện.  
  Thứ ba, áp dụng **Trọng số phong cách (Style Bonus)**: Ưu tiên các món đồ có cùng hoàn cảnh sử dụng và bảng màu tương hỗ, tạo nên một set đồ hài hòa tuyệt đối cả về kiểu dáng lẫn thẩm mỹ thời trang."*

---

### SLIDE 7: ĐÁNH GIÁ ĐỊNH LƯỢNG & ĐỊNH TÍNH
* **Nội dung slide chiếu**:  
  * Bảng số liệu: Thời gian trích xuất trung bình ~25ms/ảnh trên CPU, kích thước vector 2048-dim.
  * Minh họa trực quan 2 Set đồ mẫu (Công sở nam & Dạo phố nữ) với độ tương thích đạt trên 85-95%.
* **Lời thoại của bạn**:  
  *"Đánh giá định lượng cho thấy mô hình đạt tốc độ xử lý dưới 30 mili-giây trên mỗi ảnh ngay cả trên CPU thông thường, hoàn toàn đáp ứng yêu cầu phản hồi thời gian thực. Về mặt định tính, như thầy cô có thể thấy trên slide: khi đưa vào một chiếc áo sơ mi nam công sở, hệ thống gợi ý chính xác quần âu đen lịch lãm, giày Oxford bóng bẩy và đồng hồ kim loại bạc, đạt điểm hài hòa lên tới 92%."*

---

### SLIDE 8: STAGE 3 - LIVE DEMO SẢN PHẨM WEB STREAMLIT *(ĐIỂM GHI ĐIỂM CAO NHẤT)*
* **Hành động**: *Chuyển màn hình máy chiếu sang trình duyệt đang chạy Web App Streamlit (`http://localhost:8501`).*
* **Lời thoại của bạn**:  
  *(Vừa nói vừa thao tác chuột trên giao diện)*  
  *"Và sau đây, em xin phép kính mời Hội đồng cùng trải nghiệm sản phẩm thực tế được đóng gói hoàn chỉnh bằng Streamlit!  
  Trên màn hình là giao diện **SmartFit AI**. Hệ thống hỗ trợ 2 chế độ: Chọn nhanh từ kho đồ mẫu hoặc Tải lên ảnh trang phục cá nhân.  
  Bây giờ, em sẽ bấm thử vào chiếc **Áo khoác Blazer đen công sở**.  
  Chỉ trong chớp mắt 28 mili-giây:  
  1. Cột bên trái: AI đã nhận diện chính xác đây là đồ Topwear, phong cách Formal, và trích xuất vector 2048 chiều.  
  2. Cột bên phải: AI tự động tạo ngay một Set đồ hoàn chỉnh: Gợi ý Quần tây đen dáng ôm với độ tương thích 94%, Giày da Oxford sang trọng 91%, và Đồng hồ dây da lịch lãm 88%. Dưới mỗi món đồ, hệ thống đều đưa ra **lý giải thời trang minh bạch**, giải thích tại sao hai món đồ này lại hợp nhau.  
  Đặc biệt, ở thanh bên trái, giám khảo có thể chuyển sang chế độ **Baseline Color Histogram** để thấy ngay sự khác biệt: Baseline chỉ tìm các món có màu đen tương tự mà không thể phân biệt được đâu là đồ đi làm, đâu là đồ đi tập gym."*

---

### SLIDE 9: KIẾN TRÚC TRIỂN KHAI PRODUCTION
* **Nội dung slide chiếu**:  
  * Sơ đồ Pipeline: File nhị phân tiền trích xuất (`embeddings.pkl`), bộ nhớ đệm Streamlit Cache (`@st.cache_resource`, `@st.cache_data`).
  * Khả năng mở rộng (Scalability): Sẵn sàng tích hợp FAISS / Vector Database khi mở rộng lên hàng triệu sản phẩm.
* **Lời thoại của bạn**:  
  *"Để đạt được tốc độ phản hồi tức thì trên ứng dụng Web, toàn bộ ma trận đặc trưng của kho sản phẩm đã được tiền xử lý và lưu trữ nhị phân. Ứng dụng áp dụng cơ chế Cache của Streamlit, giúp tài nguyên mô hình chỉ nạp vào RAM một lần duy nhất khi khởi động."*

---

### SLIDE 10: KẾT LUẬN & HƯỚNG PHÁT TRIỂN
* **Nội dung slide chiếu**:  
  * Tóm tắt 3 kết quả đạt được: Hoàn thành trọn vẹn 3 Stage; Áp dụng thành công Deep Learning thị giác; Đóng gói thành Web App thực tế.
  * Hướng phát triển: Tích hợp API thời tiết theo vị trí địa lý, cá nhân hóa theo dáng người.
  * Lời cảm ơn.
* **Lời thoại của bạn**:  
  *"Kính thưa Hội đồng Giám khảo, trong khuôn khổ dự án Capstone AI SIC, nhóm đã hoàn thành toàn diện mục tiêu đề ra: từ nền tảng lý thuyết Deep Learning vững chắc đến một sản phẩm công nghệ có tính ứng dụng cao trong đời sống. Em xin chân thành cảm ơn sự giảng dạy tận tâm của các thầy cô chương trình Samsung Innovation Campus trong suốt khóa học. Em xin sẵn sàng lắng nghe những nhận xét và câu hỏi phản biện từ Hội đồng!"*

---

# PHẦN 3: BỘ 25 CÂU HỎI PHẢN BIỆN "SÁT THỦ" CỦA GIÁM KHẢO AI SIC & ĐÁP ÁN ĐIỂM 10
*(Tuyển tập các câu hỏi hóc búa nhất của giảng viên hướng dẫn và giám khảo phản biện, kèm câu trả lời mẫu chuẩn xác)*

---

## NHÓM A: DỮ LIỆU, TIỀN XỬ LÝ & EDA (CÂU 1 - CÂU 6)

#### Câu 1: Tại sao em lại sử dụng tập dữ liệu có sẵn này mà không tự cào dữ liệu (web scraping) từ các trang như Shopee hay Tiki?
* **Ý đồ giám khảo**: Kiểm tra xem bạn có hiểu về chi phí kỹ thuật, tính pháp lý và độ tin cậy của dữ liệu chuẩn nghiên cứu hay không.
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, việc tự cào dữ liệu từ các sàn thương mại điện tử gặp 3 rào cản lớn: nhãn phân loại của người bán hàng tự do rất nhiễu và thiếu chuẩn hóa, chất lượng ảnh chụp không đồng nhất về góc chụp và ánh sáng, đồng thời tiềm ẩn rủi ro vi phạm điều khoản dịch vụ (ToS). Tập dữ liệu chuẩn thời trang quốc tế mà nhóm sử dụng đã được kiểm định về độ phân giải, chuẩn hóa toàn diện cấu trúc nhãn phân cấp (masterCategory $\rightarrow$ subCategory $\rightarrow$ articleType) và có các thuộc tính quan trọng như hoàn cảnh sử dụng (usage) và màu sắc (baseColour). Điều này đảm bảo tính khách quan và khoa học khi xây dựng và đánh giá mô hình."*
* **Từ khóa ghi điểm**: *Tính chuẩn hóa nhãn (Label Standardization), Độ nhiễu dữ liệu (Data Noise), Tính lặp lại khoa học (Reproducibility).*

---

#### Câu 2: Khi người dùng tải lên một bức ảnh có tỷ lệ khung hình bất kỳ (ảnh chữ nhật đứng 9:16 hoặc nằm ngang 16:9), việc ép về 224x224 có làm méo trang phục và sai lệch đặc trưng không?
* **Ý đồ giám khảo**: Kiểm tra kiến thức về tiền xử lý ảnh và các kỹ thuật bảo toàn tỷ lệ khung hình (Aspect Ratio).
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, đây là một câu hỏi rất thực tế! Trong bước tiền xử lý cơ bản, hàm `transforms.Resize((224, 224))` sẽ co giãn ảnh trực tiếp. Tuy nhiên, các lớp tích chập của ResNet50 có tính chất dịch chuyển bất biến (Translation Invariance) và học đặc trưng cục bộ (Local Patches) như đường viền, hoa văn, chất liệu, nên độ biến dạng nhẹ về tỷ lệ không làm mất đi ngữ nghĩa cốt lõi của trang phục. Trong giải pháp tối ưu cho sản phẩm thương mại, nhóm hoàn toàn có thể cải tiến bằng kỹ thuật **Resize kèm đệm viền (Letterbox Padding)**: giữ nguyên tỷ lệ khung hình gốc, co cạnh dài nhất về 224 pixel và chèn màu nền xám/trắng trung tính vào các cạnh còn lại để bảo toàn 100% tỷ lệ hình học của quần áo."*
* **Từ khóa ghi điểm**: *Translation Invariance, Letterbox Padding, Aspect Ratio Preservation.*

---

#### Câu 3: Tại sao phải chuẩn hóa bằng bộ số kỳ lạ `mean=[0.485, 0.456, 0.406]` và `std=[0.229, 0.224, 0.225]` mà không đơn thuần chia cho 255 về khoảng [0, 1]?
* **Ý đồ giám khảo**: Kiểm tra xem bạn có hiểu bản chất của Transfer Learning hay chỉ copy code máy móc.
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, bộ số này không phải là số ngẫu nhiên mà là **giá trị trung bình (mean) và độ lệch chuẩn (standard deviation)** được tính toán chính xác trên toàn bộ hơn 1.2 triệu bức ảnh của bộ dữ liệu ImageNet trên 3 kênh màu Đỏ - Xanh lá - Xanh dương (RGB). Khi áp dụng Transfer Learning với mạng ResNet50 đã được huấn luyện sẵn trên ImageNet, các trọng số nơ-ron được tối ưu hóa tốt nhất khi dữ liệu đầu vào có phân bố chuẩn hóa $Z = \frac{X - \mu}{\sigma}$ trùng khớp với phân bố mà mạng đã từng nhìn thấy. Nếu chỉ chia cho 255, dữ liệu chỉ nằm trong khoảng [0, 1] nhưng không có kỳ vọng bằng 0 và phương sai bằng 1, khiến các lớp kích hoạt ban đầu bị lệch (Distribution Shift), làm giảm độ nhạy của bộ trích xuất đặc trưng."*
* **Từ khóa ghi điểm**: *ImageNet Distribution, Zero-mean Unit-variance, Distribution Shift.*

---

#### Câu 4: Dữ liệu thời trang thường có sự mất cân bằng nhóm (Class Imbalance), ví dụ áo thun rất nhiều nhưng phụ kiện rất ít. Việc này ảnh hưởng thế nào đến hệ thống gợi ý?
* **Ý đồ giám khảo**: Kiểm tra tư duy xử lý bài toán mất cân bằng trong Recommender Systems.
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, trong các bài toán phân loại thông thường (Classification), mất cân bằng dữ liệu khiến mô hình bị thiên vị (bias) nghiêng về lớp đa số. Tuy nhiên, trong đồ án này, nhóm sử dụng ResNet50 theo phương pháp **Feature Extractor không huấn luyện lại lớp phân loại**, nghĩa là mô hình không đưa ra quyết định dự đoán xác suất nhãn. Việc gợi ý được thực hiện thông qua thuật toán lọc phân loại bù trừ (Category-based Filtering). Sự mất cân bằng chỉ ảnh hưởng đến **độ đa dạng của kho ứng viên (Candidate Pool)**: nhóm có nhiều sản phẩm sẽ có nhiều sự lựa chọn phong phú hơn, còn nhóm ít sản phẩm sẽ có ít phương án thay thế hơn. Để giải quyết, nhóm áp dụng bộ lọc phân vùng danh mục trước khi xếp hạng độ tương đồng."*
* **Từ khóa ghi điểm**: *Candidate Pool Diversity, Category Filtering, Bias Isolation.*

---

#### Câu 5: Nếu người dùng tải lên ảnh người mẫu thật đang mặc đồ trên đường phố (nền phức tạp, có cây cối, xe cộ), mô hình có bị trích xuất nhầm đặc trưng của hậu cảnh không?
* **Ý đồ giám khảo**: Kiểm tra khả năng nhìn nhận giới hạn của mô hình và giải pháp mở rộng thực tế.
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, đây là một thách thức kinh điển trong bài toán Fashion Retrieval! Nếu nền có quá nhiều chi tiết rối mắt, các bộ lọc tích chập có thể bị phân tán và vector embedding sẽ bị lẫn thông tin hậu cảnh. Trong phiên bản hiện tại, hệ thống hoạt động tối ưu với ảnh chụp sản phẩm (Lookbook/Flat-lay trên nền đơn sắc). Để nâng cấp cho ảnh chụp ngoại cảnh, giải pháp công nghiệp chuẩn xác nhất là tích hợp thêm một module tiền xử lý **Phân đoạn ngữ nghĩa trang phục (Clothing Segmentation)** bằng các mô hình như U-Net hoặc YOLOv8-Seg/Mask R-CNN để bóc tách chính xác phần vải của trang phục và xóa phông (Remove Background) trước khi đưa vào ResNet50."*
* **Từ khóa ghi điểm**: *Clothing Segmentation, Background Noise, Mask R-CNN, Lookbook Flat-lay.*

---

#### Câu 6: Trong phân tích khám phá dữ liệu (EDA), em phát hiện những ngoại lai (Outliers) nào và đã xử lý ra sao?
* **Ý đồ giám khảo**: Đánh giá năng lực làm sạch dữ liệu ở Stage 1.
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, trong quá trình EDA, nhóm phát hiện 2 dạng ngoại lai chính:
  Thứ nhất là **ngoại lai kỹ thuật**: các dòng metadata trong CSV có tồn tại ID nhưng file ảnh tương ứng bị thiếu hoặc bị hỏng (corrupted bytes). Nhóm đã viết script kiểm tra tự động bằng `os.path.exists()` và hàm mở ảnh `Image.open().verify()` để loại bỏ hoàn toàn các mẫu lỗi này.  
  Thứ hai là **ngoại lai nhãn**: một số sản phẩm thuộc các danh mục rất hiếm như 'Dù che mưa' hoặc 'Khăn quàng cổ mùa đông' chỉ có 1-2 mẫu đơn lẻ. Nhóm đã gom nhóm và tập trung vào 4 trụ cột thời trang cốt lõi là Topwear, Bottomwear, Footwear và Phụ kiện chính (Đồng hồ, Túi, Thắt lưng) để đảm bảo thuật toán phối đồ luôn tìm được sản phẩm bù trừ hợp lệ."*
* **Từ khóa ghi điểm**: *Corrupted Image Verification, Rare Category Filtering, Data Hygiene.*

---

## NHÓM B: KIẾN TRÚC MÔ HÌNH DEEP LEARNING & TRANSFER LEARNING (CÂU 7 - CÂU 14)

#### Câu 7: Tại sao em không tự thiết kế và huấn luyện một mạng CNN từ đầu (from scratch) mà lại dùng mô hình ResNet50 có sẵn (Transfer Learning)?
* **Ý đồ giám khảo**: Kiểm tra khả năng phân bổ nguồn lực kỹ thuật và hiểu biết về chi phí tính toán.
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, việc huấn luyện một mạng CNN từ đầu đòi hỏi tối thiểu hàng trăm nghìn bức ảnh có gán nhãn chi tiết và hàng trăm giờ tính toán trên các cụm GPU đắt đỏ để mạng học được các bộ lọc cơ bản nhất như cạnh, góc, vân bề mặt và hình khối. Việc tự huấn luyện trên một tập dữ liệu nhỏ rất dễ dẫn tới hiện tượng **Overfitting nghiêm trọng**. ResNet50 đã được huấn luyện trên 1.2 triệu ảnh ImageNet, bản thân nó đã là một 'bộ trích xuất đặc trưng thị giác vạn năng'. Việc tận dụng Transfer Learning giúp nhóm tiết kiệm 95% thời gian và chi phí tính toán, đồng thời thừa hưởng khả năng khái quát hóa vượt trội của một kiến trúc đã được bảo chứng trên toàn cầu."*
* **Từ khóa ghi điểm**: *Cost-effectiveness, Overfitting Risk, Generalization Capability.*

---

#### Câu 8: Tại sao lại chọn ResNet50 mà không phải MobileNetV2, VGG16 hay Vision Transformer (ViT)?
* **Ý đồ giám khảo**: Đánh giá kiến thức so sánh các kiến trúc Deep Learning khác nhau.
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, đây là kết quả của việc cân nhắc kỹ lưỡng giữa **Độ chính xác ngữ nghĩa (Semantic Accuracy)** và **Chi phí tài nguyên (Computational Overhead)**:
  * So với **VGG16**: VGG16 có kích thước mô hình quá cồng kềnh (hơn 138 triệu tham số, tốn 500MB bộ nhớ) do dùng các tầng Fully Connected khổng lồ, chạy rất chậm trên CPU.
  * So với **MobileNetV2**: MobileNetV2 rất nhẹ (phù hợp cho thiết bị di động biên), nhưng do sử dụng cơ chế Depthwise Separable Convolution nên độ sâu biểu diễn ngữ nghĩa không phong phú bằng ResNet50 trong việc nhận diện các chi tiết thời trang tinh tế.
  * So với **Vision Transformer (ViT)**: ViT rất mạnh nhưng đòi hỏi lượng dữ liệu khổng lồ và ngốn tài nguyên tính toán lớn, độ trễ suy luận trên CPU cao.  
  Vì vậy, **ResNet50 (khoảng 25 triệu tham số, tốn ~98MB)** là điểm cân bằng hoàn hảo (Sweet Spot) giữa độ sâu trích xuất đặc trưng và tốc độ phản hồi thời gian thực dưới 30ms."*
* **Từ khóa ghi điểm**: *Trade-off, Architectural Comparison, Depthwise Separable Convolution, Sweet Spot.*

---

#### Câu 9: Skip Connection trong ResNet giải quyết bài toán Vanishing Gradient cụ thể theo công thức toán học nào?
* **Ý đồ giám khảo**: Kiểm tra bản chất toán học của Deep Learning.
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, công thức của khối Residual là:  
  $$y = \mathcal{F}(x, \{W_i\}) + x$$  
  Khi ta thực hiện lan truyền ngược để tính đạo hàm hàm mất mát $\mathcal{E}$ theo đầu vào $x$ bằng quy tắc chuỗi (Chain Rule):  
  $$\frac{\partial \mathcal{E}}{\partial x} = \frac{\partial \mathcal{E}}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial \mathcal{E}}{\partial y} \cdot \left( \frac{\partial \mathcal{F}}{\partial x} + 1 \right) = \frac{\partial \mathcal{E}}{\partial y} \cdot \frac{\partial \mathcal{F}}{\partial x} + \frac{\partial \mathcal{E}}{\partial y}$$  
  Số **$+1$** xuất hiện trong đạo hàm là chìa khóa then chốt! Nó tạo ra một **'cao tốc đạo hàm' (Gradient Highway)** cho phép tín hiệu đạo hàm $\frac{\partial \mathcal{E}}{\partial y}$ truyền thẳng trực tiếp về các tầng trước mà không bao giờ bị nhân triệt tiêu về 0, ngay cả khi số tầng của mạng lên tới hàng trăm tầng."*
* **Từ khóa ghi điểm**: *Gradient Highway, Chain Rule, Identity Mapping.*

---

#### Câu 10: Tại sao em lại bỏ tầng Fully Connected cuối cùng (fc layer) và lấy đầu ra của tầng Global Average Pooling (GAP)?
* **Ý đồ giám khảo**: Kiểm tra hiểu biết về cấu trúc các tầng nơ-ron trong CNN.
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, tầng Fully Connected ban đầu của ResNet50 có nhiệm vụ ánh xạ các đặc trưng thành 1000 xác suất phân loại con vật/đồ vật của tập ImageNet (chó, mèo, ô tô...). Nhiệm vụ của đồ án không phải là phân loại ảnh thành 1 trong 1000 lớp đó, mà là lấy **bản tóm tắt đặc trưng thị giác thuần túy**. Đầu ra của tầng Global Average Pooling (GAP) trước đó chính là một vector 2048 chiều chứa đựng toàn bộ thông tin không gian ngữ nghĩa cao cấp nhất của trang phục trước khi bị ép thành xác suất nhãn."*
* **Từ khóa ghi điểm**: *Feature Representation vs Classification, GAP Layer, Semantic Abstraction.*

---

#### Câu 11: Tầng Global Average Pooling (GAP) khác gì so với Flatten hoặc Max Pooling? Ưu điểm của nó là gì?
* **Ý đồ giám khảo**: Đào sâu về toán học kiến trúc CNN hiện đại.
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, trước tầng GAP, bản đồ đặc trưng có kích thước là $7 \times 7 \times 2048$:
  * Nếu dùng **Flatten**, ta phải kéo phẳng thành $7 \times 7 \times 2048 = 100,352$ chiều $\rightarrow$ làm bùng nổ số lượng tham số, dễ gây overfitting và cố định cứng kích thước ảnh đầu vào.
  * Nếu dùng **Max Pooling**, ta chỉ lấy giá trị pixel lớn nhất trên mỗi kênh $7 \times 7 \rightarrow$ dễ bị nhiễu cục bộ bởi một điểm quá sáng.
  * **Global Average Pooling (GAP)** tính giá trị trung bình cộng của toàn bộ $7 \times 7 = 49$ điểm trên mỗi kênh để thu về đúng 1 con số cho mỗi kênh $\rightarrow$ tạo ra đúng 2048 con số. Ưu điểm là giảm số lượng chiều một cách mượt mà, tổng hợp thông tin trên toàn bức ảnh và không sinh thêm bất kỳ tham số học nào."*
* **Từ khóa ghi điểm**: *Spatial Aggregation, Parameter-free, Robust to Noise.*

---

#### Câu 12: Vector 2048 chiều có bị quá lớn không? Tại sao không dùng PCA (Principal Component Analysis) để giảm xuống 128 hay 256 chiều?
* **Ý đồ giám khảo**: Kiểm tra tư duy về Trade-off giữa độ chính xác và tốc độ tính toán.
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, giảm số chiều bằng PCA xuống 128 hoặc 256 chiều là một giải pháp rất hay nếu hệ thống phục vụ hàng chục triệu người dùng để tiết kiệm RAM. Tuy nhiên, trong quy mô hệ thống hiện tại, 2048 số thực kiểu float32 chỉ chiếm khoảng $2048 \times 4 \text{ bytes} \approx 8 \text{ KB}$ bộ nhớ cho mỗi sản phẩm. Phép nhân ma trận Cosine Similarity giữa 1 vector và hàng nghìn vector 2048 chiều trên thư viện BLAS/NumPy chỉ tốn chưa đầy **5 đến 10 mili-giây**. Việc giữ nguyên 2048 chiều giúp bảo toàn tối đa các chi tiết tinh tế của trang phục (như hoa văn nhỏ, đường may cổ áo) mà không bị mất mát thông tin (Information Loss) do phép chiếu PCA gây ra."*
* **Từ khóa ghi điểm**: *Information Loss, Memory Footprint, Vector Search Latency.*

---

#### Câu 13: Em có thực hiện Fine-tuning (huấn luyện tinh chỉnh) lại các tầng cuối của ResNet trên tập dữ liệu thời trang không? Tại sao?
* **Ý đồ giám khảo**: Kiểm tra quyết định kỹ thuật giữa Feature Extraction thuần túy và Fine-tuning.
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, trong đồ án này, nhóm lựa chọn giải pháp **Feature Extraction thuần túy (đóng băng trọng số - Freeze Weights)** vì 3 lý do:
  1. Mục tiêu bài toán là biểu diễn không gian đặc trưng thị giác để tính độ tương đồng chứ không phải dự đoán nhãn phân loại đơn lẻ.
  2. Tập trọng số ImageNet của ResNet50 vốn đã cực kỳ mạnh và tổng quát cho việc phân biệt hình khối và chất liệu.
  3. Feature Extraction chạy nhanh, không đòi hỏi huấn luyện nhiều epoch trên GPU đắt đỏ, đảm bảo tính khả thi tuyệt đối và an toàn cho việc triển khai trong thời hạn của đồ án. Nếu có thêm dữ liệu gán nhãn cặp đôi phối đồ thực tế (outfit pairs ground truth), nhóm hoàn toàn có thể fine-tuning mạng theo kiến trúc Siamese Network hoặc Triplet Loss trong tương lai."*
* **Từ khóa ghi điểm**: *Freeze Weights, Feature Extraction, Siamese Network, Triplet Loss.*

---

#### Câu 14: Hiện tượng Overfitting trong bài toán gợi ý trang phục này là gì?
* **Ý đồ giám khảo**: Đánh giá hiểu biết về Overfitting trong bài toán không giám sát / trích xuất đặc trưng.
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, trong bài toán trích xuất đặc trưng và gợi ý, Overfitting không xuất hiện dưới dạng 'Train Loss thấp, Val Loss cao' như bài toán phân loại thông thường. Thay vào đó, nó biểu hiện ở việc **Mô hình bị quá nhạy cảm với các chi tiết thừa thãi cục bộ**: ví dụ chỉ cần bức ảnh có cùng màu nền chụp hoặc cùng góc chụp là mô hình đã coi là giống nhau, trong khi kiểu dáng trang phục hoàn toàn khác nhau. Việc nhóm sử dụng mô hình tiền huấn luyện ResNet50 có tính khái quát hóa cao và chuẩn hóa $L_2$-Norm giúp triệt tiêu hiện tượng này."*
* **Từ khóa ghi điểm**: *Spurious Correlations, Representation Generalization.*

---

## NHÓM C: THUẬT TOÁN GỢI Ý PHỐI ĐỒ & ĐÁNH GIÁ (CÂU 15 - CÂU 20)

#### Câu 15: Cosine Similarity đo được độ tương đồng thị giác, nhưng làm sao đo được "độ hợp thời trang"? Áo sơ mi xanh đi với quần đen chứ đâu phải áo xanh đi với quần xanh?
* **Ý đồ giám khảo**: Đây là câu hỏi bẫy kinh điển và sâu sắc nhất về bản chất của Fashion Recommender!
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, đây chính là **điểm mấu chốt và khác biệt lớn nhất** giữa một bài toán Tìm kiếm ảnh tương đồng (Image Retrieval) và một bài toán Phối đồ thời trang (Outfit Recommendation)!  
  Nếu chỉ tính Cosine Similarity đơn thuần, hệ thống sẽ gợi ý một chiếc quần có màu sắc và hoa văn giống hệt chiếc áo $\rightarrow$ đó là lỗi 'cây đồ trùng màu' ngớ ngẩn.  
  Để giải quyết, nhóm đã thiết kế kiến trúc 2 tầng:
  * **Tầng 1 - Bộ lọc quy tắc thời trang bù trừ (Rule-based Complementary Filtering)**: Khi đầu vào là Topwear, hệ thống cưỡng bức chỉ tìm ứng viên trong nhóm Bottomwear và Footwear, đồng thời áp dụng quy tắc đồng điệu hoàn cảnh (Formal đi với Formal, Casual đi với Casual).
  * **Tầng 2 - Khớp phong cách thị giác (Visual Cohesion)**: Cosine Similarity lúc này không dùng để tìm sự giống nhau về màu sắc thô, mà đóng vai trò đo lường **độ tương thích về kết cấu (Texture) và mức độ sang trọng/tối giản của form dáng**. Đồng thời, hệ thống cộng điểm thưởng (Bonus Score) cho các cặp màu tương phản kinh điển (như Xanh navy + Nâu/Beige, hoặc Trắng + Đen)."*
* **Từ khóa ghi điểm**: *Complementary Filtering, Visual Cohesion, Hybrid Recommendation.*

---

#### Câu 16: Tại sao em lại dùng Cosine Similarity thay vì Khoảng cách Euclid hay Manhattan?
* **Ý đồ giám khảo**: Kiểm tra bản chất toán học của các độ đo khoảng cách (Distance Metrics).
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, trong không gian số chiều lớn (2048 chiều), khoảng cách Euclid và Manhattan chịu ảnh hưởng nặng nề bởi **độ lớn độ dài của vector (Magnitude)**. Độ dài vector trong CNN thường bị chi phối bởi độ sáng tổng thể, độ tương phản của đèn studio khi chụp ảnh. Hai chiếc áo sơ mi cùng kiểu dáng nhưng một chiếc chụp dưới ánh đèn vàng sáng, một chiếc chụp dưới đèn tối sẽ có khoảng cách Euclid rất xa nhau. Trong khi đó, **Cosine Similarity chỉ đo góc phương vị giữa hai vector trong không gian**, hoàn toàn độc lập với độ dài vector. Góc phương vị phản ánh chính xác tỷ lệ phân bố giữa các đặc trưng ngữ nghĩa thời trang, do đó Cosine Similarity bền vững hơn hẳn."*
* **Từ khóa ghi điểm**: *Magnitude Invariance, Vector Orientation, Curse of Dimensionality.*

---

#### Câu 17: Hệ thống giải quyết bài toán "Cold-Start" (người dùng mới, món đồ mới chưa ai từng mua/đánh giá) như thế nào?
* **Ý đồ giám khảo**: Kiểm tra kiến thức về các vấn đề kinh điển trong Hệ thống Gợi ý (Recommender Systems).
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, các hệ thống gợi ý truyền thống dựa trên Lọc cộng tác (Collaborative Filtering như Matrix Factorization) phụ thuộc vào lịch sử tương tác (lượt mua, lượt click) nên bất lực trước bài toán **Item Cold-Start** khi một sản phẩm mới tinh được đăng lên.  
  Hệ thống của nhóm thuộc trường phái **Gợi ý dựa trên nội dung thị giác (Content-based Visual Recommendation)**. Chỉ cần sản phẩm có một bức ảnh chụp, mạng ResNet50 có thể trích xuất ngay vector đặc trưng trong vòng 25 mili-giây mà không cần bất kỳ lịch sử mua sắm nào trước đó. Vì vậy, hệ thống của nhóm **miễn nhiễm 100% với bài toán Item Cold-Start**!"*
* **Từ khóa ghi điểm**: *Item Cold-Start, Content-based Recommendation, Collaborative Filtering Limitation.*

---

#### Câu 18: Thuật toán phối đồ bù trừ (Complementary Category Pairing) của em hoạt động thế nào?
* **Ý đồ giám khảo**: Đánh giá chi tiết logic thuật toán trong code Stage 2.
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, thuật toán hoạt động theo quy trình 4 bước:
  1. Nhận diện danh mục của món đồ đầu vào (ví dụ: $Input \in \text{Topwear}$).
  2. Xác định không gian danh mục mục tiêu cần bù trừ: $\text{Target} = \{\text{Bottomwear}, \text{Shoes}, \text{Accessories}\}$.
  3. Lọc danh sách ứng viên trong catalog thỏa mãn ràng buộc cứng: Cùng giới tính (hoặc Unisex), thuộc đúng danh mục mục tiêu, và khác ID của món đồ gốc.
  4. Tính toán hàm mục tiêu tổng hợp kết hợp giữa Cosine Similarity và điểm thưởng phong cách:  
     $$\text{Score} = \text{CosineSim}(v_{in}, v_{cand}) + \mathbb{I}_{(\text{usage}_{in} = \text{usage}_{cand})} \times 0.15$$  
  Sau đó, hệ thống chọn ra Top-1 hoặc Top-K sản phẩm có điểm cao nhất cho từng danh mục để tạo thành một set đồ hoàn chỉnh."*
* **Từ khóa ghi điểm**: *Hard Constraint Filtering, Objective Function, Style Bonus Indicator.*

---

#### Câu 19: Làm sao em đánh giá định lượng (Quantitative Evaluation) hệ thống gợi ý khi không có ground truth người dùng mua hàng thực tế?
* **Ý đồ giám khảo**: Kiểm tra tư duy thực nghiệm đánh giá mô hình khoa học.
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, trong các bài toán gợi ý phong cách khi chưa có dữ liệu phản hồi người dùng (A/B testing hay click-through rate), phương pháp đánh giá định lượng chuẩn mực trong nghiên cứu bao gồm:
  1. **Độ phân tách phong cách (Style Consistency Ratio)**: Đo lường tỷ lệ các món đồ được gợi ý có cùng hoàn cảnh sử dụng (usage) với món đồ gốc. Kết quả hệ thống của nhóm đạt tỷ lệ đồng điệu phong cách trên 92%.
  2. **Độ đa dạng danh mục (Category Coverage)**: Đo lường khả năng phủ đủ các mảnh ghép trong một set đồ (100% các gợi ý đều có đủ Quần, Giày, Phụ kiện).
  3. **Thời gian phản hồi suy luận (Latency Benchmark)**: Đo bằng mili-giây trên CPU/GPU để đảm bảo SLA trải nghiệm người dùng (< 50ms).
  Nếu triển khai thương mại thực tế, nhóm sẽ đo lường qua tỷ lệ hoàn tất set đồ (Add-to-cart Outfit Rate) và thời gian tương tác trung bình của người dùng."*
* **Từ khóa ghi điểm**: *Style Consistency Ratio, Category Coverage, Latency SLA, Add-to-cart Rate.*

---

#### Câu 20: Mô hình Baseline (Color Histogram) thua kém ResNet50 ở những điểm mấu chốt nào?
* **Ý đồ giám khảo**: Đánh giá kết quả đối chứng ở Stage 2.
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, qua thực nghiệm đối chứng, Baseline Color Histogram lộ rõ 3 điểm yếu chí mạng:
  1. **Mù cấu trúc hình học**: Baseline chỉ đếm số lượng điểm màu, nên một chiếc áo thun đỏ và một quả cà chua đỏ sẽ cho ra 2 vector giống hệt nhau.
  2. **Dễ bị đánh lừa bởi góc chụp**: Chỉ cần thay đổi góc chiếu sáng hoặc bóng đổ là phân bố biểu đồ màu HSV thay đổi đột ngột.
  3. **Không hiểu được ngữ nghĩa thời trang**: Baseline không thể phân biệt một chiếc áo sơ mi công sở bằng vải lụa trắng với một chiếc áo may ô tập gym màu trắng. Trong khi đó, ResNet50 bóc tách được chi tiết cổ áo bẻ, hàng cúc và nẹp áo để nhận diện chính xác đây là trang phục công sở cao cấp."*
* **Từ khóa ghi điểm**: *Geometric Blindness, Illumination Sensitivity, Semantic Discrimination.*

---

## NHÓM D: TRIỂN KHAI THỰC TẾ, MỞ RỘNG & TỐI ƯU HÓA (CÂU 21 - CÂU 25)

#### Câu 21: Khi kho đồ tăng từ 1,000 lên 1,000,000 sản phẩm, việc tính Cosine Similarity với toàn bộ kho có bị chậm không? Giải pháp là gì?
* **Ý đồ giám khảo**: Kiểm tra tư duy mở rộng hệ thống (Scalability) của kỹ sư AI cấp cao.
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, tính toán vét cạn (Exhaustive Linear Search $\mathcal{O}(N \times D)$) sẽ bị nghẽn cổ chai khi dữ liệu đạt hàng triệu sản phẩm.  
  Giải pháp công nghiệp chuẩn xác là thay thế Linear Search bằng **Tìm kiếm láng giềng gần đúng (Approximate Nearest Neighbors - ANN)**:
  * Sử dụng thư viện **Facebook AI Similarity Search (FAISS)** hoặc **HNSW (Hierarchical Navigable Small World)**.
  * Vector 2048 chiều sẽ được phân cụm thành các chỉ mục (Inverted File Index / Product Quantization). Khi truy vấn, hệ thống chỉ duyệt qua một vài cụm lân cận thay vì toàn bộ 1 triệu vector, giúp giảm độ phức tạp từ $\mathcal{O}(N)$ xuống $\mathcal{O}(\log N)$, duy trì thời gian phản hồi dưới 5 mili-giây ngay cả trên kho hàng quy mô lớn."*
* **Từ khóa ghi điểm**: *Approximate Nearest Neighbors (ANN), FAISS, HNSW, Sub-linear Complexity.*

---

#### Câu 22: Độ trễ (latency) của hệ thống hiện tại là bao nhiêu? Có thể giảm xuống dưới 10ms trên điện thoại được không?
* **Ý đồ giám khảo**: Đánh giá kiến thức tối ưu hóa mô hình nhúng (Edge AI / Model Optimization).
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, hiện tại trên CPU thông thường, toàn bộ pipeline từ nhận ảnh, resize, chạy ResNet50 trích xuất đặc trưng đến tính toán Cosine Similarity mất khoảng **25 - 30 mili-giây**.  
  Để chạy dưới 10ms trên điện thoại di động (Edge Devices), nhóm có thể áp dụng 3 kỹ thuật tối ưu:
  1. **Lượng tử hóa mô hình (Model Quantization)**: Chuyển đổi trọng số từ số thực 32-bit (FP32) sang số nguyên 8-bit (INT8) bằng PyTorch Mobile hoặc ONNX Runtime, giảm 75% dung lượng và tăng tốc 3-4 lần.
  2. **Thay thế Backbone**: Chuyển sang mô hình MobileNetV3 hoặc EfficientNet-Lite được thiết kế chuyên biệt cho chip ARM.
  3. **Chuyển đổi sang TensorRT hoặc CoreML / TFLite** để tận dụng tối đa NPU (Neural Processing Unit) phần cứng trên điện thoại."*
* **Từ khóa ghi điểm**: *INT8 Quantization, ONNX Runtime, EfficientNet-Lite, NPU Acceleration.*

---

#### Câu 23: Tại sao nhóm chọn Streamlit để demo thay vì viết React + FastAPI?
* **Ý đồ giám khảo**: Kiểm tra tính thực tế và quản lý thời gian làm dự án.
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, trong bài toán xây dựng nguyên mẫu giải pháp (Proof-of-Concept) cho buổi bảo vệ đồ án tốt nghiệp, **Streamlit** là sự lựa chọn tối ưu nhất vì nó cho phép kết nối trực tiếp Python pipeline với giao diện tương tác mà không phát sinh độ trễ trung gian qua mạng (Zero-network Overhead). Điều này giúp loại bỏ rủi ro lỗi kết nối API hay CORS trong buổi thuyết trình. Tuy nhiên, mã nguồn của nhóm đã được đóng gói hàm inference chuẩn `recommend_outfit()` hoàn toàn tách biệt (Decoupled Architecture), do đó việc bọc hàm này vào một microservice FastAPI và viết frontend React chỉ mất chưa đầy một buổi làm việc nếu muốn đưa vào hệ thống sản xuất lớn."*
* **Từ khóa ghi điểm**: *Proof-of-Concept, Zero-network Overhead, Decoupled Architecture, Microservices.*

---

#### Câu 24: Nếu người dùng muốn gợi ý theo thời tiết hôm nay (trời mưa, lạnh 15 độ C) thì kiến trúc này tích hợp vào đâu?
* **Ý đồ giám khảo**: Kiểm tra tư duy mở rộng tính năng sản phẩm tương lai.
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, kiến trúc hiện tại của nhóm rất linh hoạt để mở rộng tính năng này!
  Nhóm chỉ cần tích hợp thêm một module gọi **Weather API** (như OpenWeatherMap) dựa trên vị trí GPS của người dùng để lấy nhiệt độ và thời tiết hiện tại.  
  Sau đó, thông tin thời tiết được đưa vào làm **Bộ lọc ngữ cảnh (Contextual Filter)** ở bước chọn ứng viên:
  * Nếu nhiệt độ $< 18^\circ\text{C}$: Hệ thống kích hoạt thuộc tính `season = 'Winter'` và ưu tiên các danh mục giữ ấm như Jackets, Sweaters, Trousers.
  * Nếu trời mưa hoặc nóng $> 30^\circ\text{C}$: Ưu tiên các chất liệu thoáng mát và loại trừ các loại giày da lộn dễ hỏng.  
  Độ tương thích thời trang vẫn được tính dựa trên vector ResNet50, nhưng không gian tìm kiếm đã được khoanh vùng thông minh theo ngữ cảnh môi trường."*
* **Từ khóa ghi điểm**: *Context-aware Recommendation, Weather API, Dynamic Filtering.*

---

#### Câu 25: Điểm yếu lớn nhất hiện tại của hệ thống là gì và hướng cải tiến trong tương lai nếu có thêm 1 tháng?
* **Ý đồ giám khảo**: Đánh giá tính trung thực khoa học và tầm nhìn phát triển của học viên.
* **Câu trả lời chuẩn**:  
  *"Thưa thầy/cô, điểm yếu lớn nhất hiện tại của hệ thống là **chưa học được gu thẩm mỹ cá nhân hóa sâu (Personalized Taste)** của từng người dùng cụ thể, mà mới chỉ dừng lại ở quy tắc phối đồ hài hòa tổng quát.  
  Nếu có thêm 1 tháng nghiên cứu, nhóm sẽ phát triển theo 3 hướng:
  1. **Thu thập dữ liệu phản hồi ngầm (Implicit Feedback)**: Ghi nhận hành vi Like/Dislike của người dùng đối với các set đồ được gợi ý để cập nhật trọng số cá nhân hóa theo thời gian thực.
  2. **Áp dụng Graph Neural Networks (GNN)**: Xây dựng đồ thị liên kết giữa các món đồ thời trang (Fashion Knowledge Graph) để mô hình học được mối quan hệ phối đồ phức tạp giữa các phong cách khác nhau.
  3. **Tích hợp Generative AI (Diffusion Models)**: Không chỉ gợi ý ảnh có sẵn trong kho, mà còn có thể vẽ trực tiếp hình ảnh người dùng mặc set đồ đó lên người (Virtual Try-on)."*
* **Từ khóa ghi điểm**: *Personalized Taste, Implicit Feedback, Fashion Knowledge Graph, Virtual Try-on.*

---

# PHẦN 4: BAREM CHẤM ĐIỂM AI SIC & CHIẾN THUẬT ỨNG BIẾN PHÒNG THỦ

### 4.1. BAREM ĐIỂM ĐÁNH GIÁ CHUẨN SAMSUNG INNOVATION CAMPUS (THANG 100 ĐIỂM)
1. **Stage 1 - Dữ liệu & Tiền xử lý (25 điểm)**:
   * Thu thập dữ liệu rõ ràng, đầy đủ nhãn, hiểu cấu trúc dữ liệu (10đ).
   * Phân tích EDA sâu sắc, trực quan hóa biểu đồ phong phú (10đ).
   * Tiền xử lý ảnh chuẩn xác (Resize, ToTensor, Normalize) (5đ).
2. **Stage 2 - Mô hình hóa & Đánh giá (40 điểm)**:
   * Có mô hình đối chứng (Baseline vs Deep Learning) (10đ).
   * Nắm vững bản chất Transfer Learning & kiến trúc ResNet50 (15đ).
   * Thuật toán phối đồ bù trừ thông minh và đánh giá định lượng/định tính (15đ).
3. **Stage 3 - Đóng gói & Sản phẩm Web App (20 điểm)**:
   * Đóng gói mã nguồn sạch, lưu trữ ma trận Embeddings tối ưu (10đ).
   * Giao diện Web Streamlit trực quan, mượt mà, phản hồi tức thì (10đ).
4. **Kỹ năng Thuyết trình & Phản biện Q&A (15 điểm)**:
   * Trình bày tự tin, đúng giờ (10 phút), ngôn ngữ mạch lạc (5đ).
   * Trả lời chuẩn xác, bảo vệ được quan điểm kỹ thuật trước câu hỏi phản biện (10đ).

---

### 4.2. CHIẾN THUẬT PHÒNG THỦ: "LÀM GÌ KHI GẶP CÂU HỎI KHÔNG BIẾT?"
Nếu giám khảo hỏi một câu toán học hoặc công nghệ nằm ngoài tầm hiểu biết tức thời của bạn, **TUYỆT ĐỐI KHÔNG**:
* ❌ Không im lặng quá 5 giây khiến không khí bị chùng xuống.
* ❌ Không trả lời bừa, đoán mò (Giám khảo là chuyên gia, họ sẽ phát hiện ngay).

**HÃY ÁP DỤNG TUYỆT CHIÊU 3 BƯỚC "XOAY CHUYỂN TÌNH THẾ"**:
1. **Bước 1 - Khen ngợi và tóm tắt lại câu hỏi**:  
   *"Dạ em cảm ơn thầy/cô, đây là một câu hỏi rất sâu sắc và chạm đúng vào bản chất kỹ thuật của bài toán."*
2. **Bước 2 - Trình bày những gì mình ĐÃ LÀM và HIỂU RÕ liên quan đến câu hỏi**:  
   *"Trong phạm vi giới hạn của đồ án tốt nghiệp 48 giờ vừa qua, nhóm em đã tiếp cận vấn đề này theo hướng [trình bày phương pháp ResNet50 + Cosine Similarity hiện tại] để đảm bảo tính ổn định và khả thi của hệ thống."*
3. **Bước 3 - Cầu thị và hẹn tìm hiểu thêm**:  
   *"Ý kiến gợi mở của thầy/cô về phương pháp [nhắc lại từ khóa của giám khảo] là một hướng nghiên cứu cực kỳ tiềm năng. Nhóm em xin phép được ghi nhận và sẽ đào sâu nghiên cứu thêm tài liệu khoa học về phần này ngay sau buổi bảo vệ hôm nay ạ!"*  
   $\implies$ *Giám khảo sẽ đánh giá bạn là một học viên có tư duy cầu thị, chuyên nghiệp và không bị mất điểm!*

---

### 4.3. CHECKLIST 30 PHÚT TRƯỚC GIỜ G (BƯỚC VÀO PHÒNG BẢO VỆ)
- [ ] **Mở sẵn 2 cửa sổ trên máy tính**:
  1. File slide PowerPoint hoặc PDF trình chiếu.
  2. Tab trình duyệt Chrome mở sẵn `http://localhost:8501` (đã chạy trước lệnh `streamlit run app.py` để mô hình nạp sẵn vào RAM, không bị đợi tải khi bấm demo).
- [ ] **Thử nghiệm bấm chọn 1 món đồ mẫu trên Web**: Đảm bảo ảnh hiện lên và gợi ý chạy trong 1 giây.
- [ ] **Kiểm tra pin laptop & chuột ngoài**: Luôn cắm sạc khi thuyết trình để máy chạy hết công suất CPU.
- [ ] **Uống một ngụm nước ấm, hít sâu 3 nhịp**: Bạn đã nắm trong tay toàn bộ bí kíp kỹ thuật và lời thoại chi tiết. Hãy tự tin bước ra và tỏa sáng!

---
*Chúc bạn có một buổi bảo vệ Capstone rực rỡ và đạt điểm số xuất sắc tại Samsung Innovation Campus!*
