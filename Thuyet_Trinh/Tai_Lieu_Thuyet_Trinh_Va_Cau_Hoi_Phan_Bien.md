# CẨM NANG THUYẾT TRÌNH BẢO VỆ CAPSTONE AI SIC & BỘ DỰ ĐOÁN 30 CÂU HỎI PHẢN BIỆN ĐIỂM 10

**Dự án**: Hệ thống Gợi ý Phối đồ Thời trang Thông minh (*Smart Fashion Outfit Recommender System*)  
**Khóa đào tạo**: Samsung Innovation Campus (AI SIC)  
**Học viên**: Đoàn Mạnh Dũng  
**Thời lượng báo cáo**: 10 – 15 Phút  

---

## MỤC LỤC TỔNG QUAN

- [PHẦN 1: KỊCH BẢN THUYẾT TRÌNH CHI TIẾT 10 PHÚT (TỪNG SLIDE)](#phan-1)
  - [Slide 1: Tiêu đề & Giới thiệu bản thân (00:00 - 00:45)](#slide-1)
  - [Slide 2: Đặt vấn đề & Mục tiêu giải pháp (00:45 - 01:45)](#slide-2)
  - [Slide 3: Tập Dữ liệu Kaggle Myntra & Phân tích EDA (01:45 - 02:45)](#slide-3)
  - [Slide 4: Pipeline Tiền xử lý & Chuẩn hóa ImageNet Z-Score (02:45 - 03:45)](#slide-4)
  - [Slide 5: Kiến trúc Mô hình Lõi: ResNet50 Transfer Learning (03:45 - 05:00)](#slide-5)
  - [Slide 6: Chuẩn hóa L2-Norm & Độ tương đồng Cosine (05:00 - 06:15)](#slide-6)
  - [Slide 7: Thực nghiệm Đối chứng: ResNet50 vs. Baseline (06:15 - 07:15)](#slide-7)
  - [Slide 8: Kết quả Kiểm thử Thực tế & Qualitative Test Cases (07:15 - 08:15)](#slide-8)
  - [Slide 9: Trình diễn Trực tiếp Web App Streamlit (Live Demo) (08:15 - 10:15)](#slide-9)
  - [Slide 10: Tổng kết Đóng góp Chuẩn 3 Stage AI SIC (10:15 - 11:00)](#slide-10)
  - [Slide 11: Hướng Phát triển Mở rộng Production (11:00 - 11:45)](#slide-11)
  - [Slide 12: Lời cảm ơn & Chuyển sang Phiên Q&A (11:45 - 12:00)](#slide-12)
- [PHẦN 2: BỘ DỰ ĐOÁN 30 CÂU HỎI PHẢN BIỆN CỦA HỘI ĐỒNG VÀ ĐÁP ÁN ĐIỂM 10](#phan-2)
  - [Chuyên đề 1: Dữ liệu & Tiền xử lý (Câu 1 - 6)](#chuyen-de-1)
  - [Chuyên đề 2: Kiến trúc Deep Learning & ResNet50 (Câu 7 - 12)](#chuyen-de-2)
  - [Chuyên đề 3: Toán học, Embeddings & Cosine Similarity (Câu 13 - 18)](#chuyen-de-3)
  - [Chuyên đề 4: Thực nghiệm, Kiểm thử & Đánh giá Đối chứng (Câu 19 - 24)](#chuyen-de-4)
  - [Chuyên đề 5: Hệ thống, Mở rộng Production & Thực tế (Câu 25 - 30)](#chuyen-de-5)
- [PHẦN 3: BẢNG TỔNG HỢP CÁC CON SỐ "VÀNG" CẦN NHỚ KHI ĐỨNG BẢO VỆ](#phan-3)

---

# PHẦN 1: KỊCH BẢN THUYẾT TRÌNH CHI TIẾT 10 PHÚT (TỪNG SLIDE)

### Quy tắc vàng khi đứng thuyết trình trước Hội đồng AI SIC:
1. **Phong thái tự tin, mắt nhìn thẳng vào Hội đồng**: Không cúi gầm mặt đọc slide. Slide chỉ là phông nền minh họa, bạn mới là trung tâm của bài báo cáo.
2. **Nói rõ ràng, tốc độ vừa phải (120 - 140 từ/phút)**: Giữ nhịp thở đều, kết thúc câu dứt khoát.
3. **Không đọc nguyên văn slide**: Slide chiếu các ý chính, bạn dùng lời nói để mở rộng bản chất kỹ thuật.

---

### Slide 1: Tiêu đề & Giới thiệu bản thân (00:00 - 00:45)
* **Hình ảnh trên màn hình**: Slide 1 Title hiển thị tên đề tài "Smart Fashion Outfit Recommender System" và tên học viên Đoàn Mạnh Dũng.
* **Lời thoại chuẩn**:
  > *"Kính thưa Quý Thầy Cô trong Hội đồng Giám khảo và các bạn học viên, em tên là Đoàn Mạnh Dũng. Hôm nay, em xin phép được đại diện bảo vệ Đồ án Tốt nghiệp Capstone khóa đào tạo Samsung Innovation Campus với đề tài: **'Hệ thống Gợi ý Phối đồ Thời trang Thông minh ứng dụng Deep Learning'**.  
  > Dự án của em tập trung giải quyết trọn vẹn 3 giai đoạn theo đúng chuẩn của AI SIC: Từ khám phá dữ liệu, nghiên cứu mô hình học sâu ResNet50 đến đóng gói sản phẩm Web tương tác thời gian thực. Sau đây, em xin phép được bắt đầu phần trình bày."*
* **Thời gian**: ~45 giây.

---

### Slide 2: Đặt vấn đề & Mục tiêu giải pháp (00:45 - 01:45)
* **Hình ảnh trên màn hình**: Slide 2 chia 2 cột: Nỗi đau thực tế vs. Mục tiêu giải pháp.
* **Lời thoại chuẩn**:
  > *"Thưa Thầy Cô, xuất phát điểm của đề tài đến từ một bài toán rất đời thường: Mỗi sáng, người tiêu dùng thường mất từ 15 đến 20 phút trước tủ quần áo với câu hỏi 'Hôm nay mặc gì?'.  
  > Trên các sàn thương mại điện tử hiện nay, hầu hết các hệ thống gợi ý chỉ dừng lại ở bài toán **Item-to-Item Similarity** — nghĩa là khi khách hàng xem một chiếc áo sơ mi xanh, hệ thống lại tiếp tục gợi ý 10 chiếc áo sơ mi xanh khác. Điều này gây nhàm chán và không giải quyết được nhu cầu mặc đẹp.  
  > Nhận thấy khoảng trống đó, mục tiêu của dự án em là xây dựng một **Động cơ Phối đồ Bù trừ (Complementary Outfit Recommendation)**: Người dùng chỉ cần cung cấp 1 món đồ gốc (ví dụ chiếc áo), AI sẽ tự động phân tích phong cách và đề xuất trọn bộ gồm Quần, Giày và Phụ kiện hòa hợp nhất với thời gian phản hồi siêu tốc dưới 30 mili-giây."*
* **Thời gian**: ~60 giây.

---

### Slide 3: Tập Dữ liệu Kaggle Myntra & Phân tích EDA (01:45 - 02:45)
* **Hình ảnh trên màn hình**: Slide 3 hiển thị thông số 1,200 sản phẩm, biểu đồ phân bố 4 nhóm và thuộc tính.
* **Lời thoại chuẩn**:
  > *"Bước vào Stage 1, để đảm bảo tính thực tiễn công nghiệp, em sử dụng tập dữ liệu thực tế từ sàn thương mại điện tử thời trang Myntra được công bố trên Kaggle.  
  > Quy mô tập dữ liệu gồm **1,200 sản phẩm thật** với hình ảnh độ phân giải cao và bảng metadata đa trường thông tin. Qua bước phân tích khám phá EDA, tập dữ liệu phân bố cân bằng thành 4 nhóm danh mục thiết yếu: Áo (Topwear) chiếm gần 45%, Giày dép chiếm 25%, Phụ kiện 17% và Quần váy chiếm 13%.  
  > Sự phân bổ này tạo tiền đề hoàn hảo để thuật toán có thể phối chéo bù trừ giữa các danh mục. 100% các mẫu dữ liệu đều được em kiểm tra toàn vẹn tệp ảnh và làm sạch các giá trị khuyết thiếu."*
* **Thời gian**: ~60 giây.

---

### Slide 4: Pipeline Tiền xử lý & Chuẩn hóa ImageNet Z-Score (02:45 - 03:45)
* **Hình ảnh trên màn hình**: Slide 4 hiển thị 3 bước: Resize 224x224, ToTensor, Z-Score Normalization.
* **Lời thoại chuẩn**:
  > *"Để chuẩn bị dữ liệu đầu vào cho mạng nơ-ron tích chập, toàn bộ pipeline tiền xử lý được em đóng gói độc lập tại file `Model/preprocess.py`.  
  > Quy trình gồm 3 bước bắt buộc:  
  > Thứ nhất, **Resize về 224 x 224 pixels** để đồng nhất kích thước đầu vào của mạng CNN.  
  > Thứ hai, **Chuyển thành Tensor số thực [0.0, 1.0]**.  
  > Và quan trọng nhất là bước thứ ba: **Chuẩn hóa Z-Score** theo giá trị Mean và Standard Deviation chuẩn của tập ImageNet. Bước này giúp đưa phân phối pixel của ảnh đầu vào khớp chính xác với miền giá trị mà các nơ-ron tích chập đã được tối ưu hóa, loại bỏ hoàn toàn hiện tượng lệch trọng số khi trích xuất đặc trưng."*
* **Thời gian**: ~60 giây.

---

### Slide 5: Kiến trúc Mô hình Lõi: ResNet50 Transfer Learning (03:45 - 05:00)
* **Hình ảnh trên màn hình**: Slide 5 hiển thị kiến trúc ResNet50, công thức Skip Connection y = F(x) + x và tầng GAP.
* **Lời thoại chuẩn**:
  > *"Sang Stage 2, em lựa chọn kiến trúc mạng nơ-ron học sâu **ResNet50** với kỹ thuật **Transfer Learning**.  
  > Tại sao lại là ResNet50 mà không phải một mạng CNN truyền thống? Điểm đột phá toán học của ResNet nằm ở khối **Residual Block** với cơ chế kết nối tắt **Skip Connection: $y = \mathcal{F}(x) + x$**. Khi tính đạo hàm lan truyền ngược, thành phần đạo hàm luôn được cộng thêm $+1$. Số $+1$ này chính là con đường cao tốc giúp tín hiệu gradient truyền thẳng về các tầng đầu tiên mà không bao giờ bị triệt tiêu, giải quyết triệt để vấn đề Vanishing Gradient ở mạng 50 tầng.  
  > Em tiến hành đóng băng toàn bộ 50 tầng tích chập để tận dụng tri thức nhận diện thị giác từ 1.28 triệu ảnh ImageNet. Sau đó, em cắt bỏ tầng phân loại cũ và khai thác tầng **Global Average Pooling (GAP)** để nén bản đồ đặc trưng thành một vector biểu diễn ngữ nghĩa **2048 chiều** cho mỗi trang phục."*
* **Thời gian**: ~75 giây.

---

### Slide 6: Chuẩn hóa L2-Norm & Độ tương đồng Cosine (05:00 - 06:15)
* **Hình ảnh trên màn hình**: Slide 6 hiển thị công thức L2-norm, Cosine Similarity và hàm mục tiêu Pairing Score.
* **Lời thoại chuẩn**:
  > *"Sau khi trích xuất vector 2048 chiều, bài toán đặt ra là làm thế nào để so sánh độ hòa hợp giữa các món đồ?  
  > Ở đây, em áp dụng **Chuẩn hóa $L_2$-Norm** để đưa mọi vector về mặt cầu đơn vị có độ dài bằng 1. Ý nghĩa thực tế của bước này là triệt tiêu sai lệch về ánh sáng: Hai bức ảnh chụp cùng một chiếc áo ở điều kiện sáng mạnh hay yếu đều có vector chuẩn hóa trùng nhau.  
  > Tiếp theo, em sử dụng **Cosine Similarity** thay vì Khoảng cách Euclid. Trong không gian 2048 chiều, khoảng cách Euclid rơi vào 'Bẫy số chiều' (Curse of Dimensionality) khiến mọi điểm đều cách đều nhau. Trong khi đó, góc phương vị Cosine phản ánh chính xác sự đồng điệu về phong cách và form dáng. Nhờ vector đã chuẩn hóa $L_2$, phép tính Cosine suy biến thành phép nhân ma trận (Dot Product) siêu tốc.  
  > Cuối cùng, hàm mục tiêu phối đồ kết hợp điểm Cosine với điểm thưởng phong cách (Style Bonus) nếu cùng hoàn cảnh sử dụng (Formal hay Casual)."*
* **Thời gian**: ~75 giây.

---

### Slide 7: Thực nghiệm Đối chứng: ResNet50 vs. Baseline (06:15 - 07:15)
* **Hình ảnh trên màn hình**: Slide 7 bảng so sánh định lượng và định tính giữa ResNet50 và Baseline HSV.
* **Lời thoại chuẩn**:
  > *"Để chứng minh tính ưu việt của giải pháp, em đã xây dựng một mô hình đối chứng **Baseline** dựa trên trích xuất màu sắc không gian HSV 512 chiều.  
  > Về mặt định lượng: Cả hai mô hình đều có tốc độ truy vấn cực nhanh dưới 1 mili-giây và ma trận lưu trữ chỉ dưới 10 Megabytes.  
  > Tuy nhiên, về mặt định tính thời trang, **ResNet50 vượt trội hoàn toàn**: Mô hình Baseline chỉ nhìn vào màu sắc nên khi chọn áo thun xám, nó chỉ gợi ý quần xám và giày xám một cách đơn điệu. Ngược lại, ResNet50 hiểu được chất liệu cotton, phom dáng thể thao, đường viền cạp quần, từ đó đề xuất set đồ phối màu tương phản hài hòa đúng chuẩn phong cách thời trang hiện đại."*
* **Thời gian**: ~60 giây.

---

### Slide 8: Kết quả Kiểm thử Thực tế & Qualitative Test Cases (07:15 - 08:15)
* **Hình ảnh trên màn hình**: Slide 8 hiển thị 2 Test Cases (Nam công sở và Nữ dạo phố) từ file `Model/test_model.py`.
* **Lời thoại chuẩn**:
  > *"Toàn bộ quy trình kiểm định được em tự động hóa trong file `Model/test_model.py`.  
  > Ở **Test Case 1**: Khi đưa vào một chiếc áo sơ mi nam xanh dương công sở, hệ thống đề xuất ngay Quần âu Peter England màu nâu sẫm với độ tương thích 84.4%, kết hợp giày da và đồng hồ nam lịch lãm. Đây chính là công thức phối màu kinh điển của trang phục văn phòng.  
  > Ở **Test Case 2**: Với chiếc áo thun dạo phố nữ họa tiết xanh lá, hệ thống phối cùng Quần short bò Jealous 21 đạt 78.5% và balo năng động.  
  > Các bài test về độ bền (Robustness) trước vector nhiễu ngẫu nhiên cũng đều vượt qua 100% không phát sinh lỗi."*
* **Thời gian**: ~60 giây.

---

### Slide 9: Trình diễn Trực tiếp Web App Streamlit (Live Demo) (08:15 - 10:15)
* **Hành động**: Chuyển cửa sổ sang trình duyệt đang chạy `streamlit run app_web/app.py`.
* **Lời thoại chuẩn**:
  > *"Kính thưa Hội đồng, để minh chứng cho khả năng ứng dụng thực tế ở Stage 3, em xin phép được trình diễn trực tiếp giao diện Web thời gian thực tại thư mục `app_web/`.  
  > *(Thao tác click)*: Em xin chọn thử một chiếc áo thun thể thao nam trong kho mẫu... Như Thầy Cô có thể thấy, chỉ trong vòng **28 mili-giây**, hệ thống đã hiển thị ngay lập tức một Set đồ hoàn chỉnh gồm Quần thể thao, Giày chạy bộ và Phụ kiện đi kèm với điểm hòa hợp trên 85%.  
  > *(Thao tác gạt nút)*: Bây giờ, em xin gạt thanh điều khiển sang mô hình **Baseline**... Thầy Cô có thể thấy rõ sự khác biệt: Baseline chỉ tìm các sản phẩm cùng tông màu đơn sắc, trong khi chuyển lại **ResNet50**, hệ thống hiểu được kiểu dáng và phối đồ vô cùng thời thượng.  
  > *(Lưu ý thêm)*: Người dùng cũng hoàn toàn có thể tải ảnh chụp từ điện thoại cá nhân lên để nhận gợi ý tức thì."*
* **Thời gian**: ~120 giây (2 phút demo).

---

### Slide 10: Tổng kết Đóng góp Chuẩn 3 Stage AI SIC (10:15 - 11:00)
* **Hình ảnh trên màn hình**: Slide 10 hiển thị tổng kết 3 Stage.
* **Lời thoại chuẩn**:
  > *"Quay trở lại với kết luận, đồ án của em đã hoàn thành trọn vẹn và xuất sắc mục tiêu đề ra:  
  > 1. Làm chủ dữ liệu: Xử lý 1,200 sản phẩm thật sạch sẽ.  
  > 2. Làm chủ công nghệ: Ứng dụng thành công Transfer Learning ResNet50 với chuẩn hóa $L_2$ và Cosine Similarity.  
  > 3. Làm chủ sản phẩm: Đóng gói pipeline phản hồi siêu tốc dưới 30ms, kiến trúc phân tách rõ ràng giữa lõi mô hình `Model/` và tầng giao diện `app_web/`."*
* **Thời gian**: ~45 giây.

---

### Slide 11: Hướng Phát triển Mở rộng Production (11:00 - 11:45)
* **Hình ảnh trên màn hình**: Slide 11 hiển thị FAISS Indexing, Weather API và Virtual Try-on.
* **Lời thoại chuẩn**:
  > *"Về hướng phát triển dài hạn khi triển khai quy mô công nghiệp:  
  > Thứ nhất, khi kho đồ mở rộng lên hàng triệu sản phẩm, em sẽ tích hợp thư viện **FAISS** để tìm kiếm láng giềng gần đúng ANN với độ trễ dưới 3ms.  
  > Thứ hai, tích hợp **API thời tiết theo GPS** để gợi ý outfit phù hợp với nhiệt độ ngoài trời.  
  > Và thứ ba, tích hợp **Mô hình Khuếch tán (Generative Diffusion)** để người dùng có thể thử đồ ảo (Virtual Try-on) trực tiếp trên cơ thể mình."*
* **Thời gian**: ~45 giây.

---

### Slide 12: Lời cảm ơn & Chuyển sang Phiên Q&A (11:45 - 12:00)
* **Hình ảnh trên màn hình**: Slide 12 hiển thị lời cảm ơn trang trọng.
* **Lời thoại chuẩn**:
  > *"Em xin chân thành cảm ơn Quý Thầy Cô trong Hội đồng Giám khảo đã chú ý lắng nghe bài báo cáo của em. Em rất mong nhận được những nhận xét và câu hỏi phản biện từ Thầy Cô để đồ án ngày càng hoàn thiện hơn. Em xin trân trọng cảm ơn!"*
* **Thời gian**: ~15 giây.

---

# PHẦN 2: BỘ DỰ ĐOÁN 30 CÂU HỎI PHẢN BIỆN CỦA HỘI ĐỒNG VÀ ĐÁP ÁN ĐIỂM 10

---

## Chuyên đề 1: Dữ liệu & Tiền xử lý (Data & Preprocessing)

#### Câu 1: Tại sao em lại chọn tập dữ liệu Kaggle Myntra mà không tự cào dữ liệu từ Shopee hoặc Tiki?
* **Ý đồ Giám khảo**: Kiểm tra tính thực tế và khả năng quản lý thời gian của bạn trong 48 giờ.
* **Đáp án điểm 10**:
  > *"Thưa Thầy Cô, việc tự cào dữ liệu (web scraping) từ các sàn thương mại điện tử như Shopee hay Tiki gặp 2 rào cản lớn: Thứ nhất là cơ chế chống cào (Anti-bot/Captcha) và thứ hai là nhãn dữ liệu rất bẩn, thiếu chuẩn hóa (ví dụ người bán tự đặt tên áo là 'áo xinh hot trend' thay vì 'Áo sơ mi').  
  > Tập dữ liệu Myntra từ Kaggle là tập dữ liệu chuẩn mực quốc tế, chứa 1,200 ảnh chụp studio có nền trắng đồng nhất và bảng metadata đa trường được phân loại bởi các chuyên gia thời trang. Với khung thời gian 48 giờ của đồ án tốt nghiệp, việc sử dụng tập dữ liệu này đảm bảo chất lượng khoa học cao nhất và tập trung toàn bộ nguồn lực vào việc tối ưu thuật toán Deep Learning."*

#### Câu 2: Tập dữ liệu 1,200 sản phẩm có bị mất cân bằng dữ liệu (Class Imbalance) không? Em giải quyết thế nào?
* **Ý đồ Giám khảo**: Kiểm tra xem bạn có nhìn thấy sự chênh lệch số lượng (Áo 45%, Quần 13%) trong lúc làm EDA không.
* **Đáp án điểm 10**:
  > *"Dạ thưa Thầy Cô, có sự chênh lệch số lượng giữa các nhóm: Topwear chiếm 44.7%, trong khi Bottomwear chiếm 12.8%.  
  > Tuy nhiên, trong bài toán Recommender System theo luật phối đồ bù trừ (Complementary Category Recommendation), sự mất cân bằng này **hoàn toàn không ảnh hưởng tiêu cực** như trong bài toán phân loại (Classification). Lý do là vì khi người dùng đưa vào một chiếc Áo (Topwear), không gian ứng viên tìm kiếm của em bị thu hẹp vào đúng nhóm Bottomwear. Thuật toán chỉ tìm chiếc quần phù hợp nhất trong 153 chiếc quần hiện có mà không bị các nhóm khác lấn át."*

#### Câu 3: Tại sao bắt buộc phải chuẩn hóa ảnh theo phân phối Z-Score ImageNet (Mean và Std) thay vì chỉ chia cho 255?
* **Ý đồ Giám khảo**: Kiểm tra kiến thức nền tảng về Computer Vision và PyTorch Transforms.
* **Đáp án điểm 10**:
  > *"Thưa Thầy Cô, phép chia cho 255 chỉ đưa pixel về khoảng `[0.0, 1.0]`. Tuy nhiên, mạng ResNet50 ban đầu được các nhà khoa học huấn luyện trên tập ImageNet với phân phối chuẩn Z-Score: trừ đi Mean `[0.485, 0.456, 0.406]` và chia cho Std `[0.229, 0.224, 0.225]`.  
  > Nếu ta không chuẩn hóa đúng Mean/Std này, dữ liệu đầu vào sẽ bị lệch phân phối (Covariate Shift). Khi đó, các tầng tích chập của ResNet50 sẽ kích hoạt sai lệch, làm suy giảm chất lượng của vector đặc trưng ngữ nghĩa."*

#### Câu 4: Nếu người dùng upload một bức ảnh có tỷ lệ khung hình rất dài hoặc rất dẹt, việc ép về 224x224 có làm méo ảnh không?
* **Ý đồ Giám khảo**: Kiểm tra khả năng xử lý tình huống thực tế khi deploy app.
* **Đáp án điểm 10**:
  > *"Thưa Thầy Cô, phép Resize trực tiếp về 224x224 có thể làm co dãn tỷ lệ hình học một chút. Tuy nhiên, mạng CNN học các đặc trưng cục bộ (Local Features) như chất liệu vải, ve cổ áo, đường cúc, họa tiết kẻ sọc. Những đặc trưng này mang tính bất biến tương đối với sự co dãn nhẹ.  
  > Trong phiên bản mở rộng tiếp theo, em sẽ bổ sung kỹ thuật **Padding giữ nguyên tỷ lệ (Letterboxing)**: chèn thêm viền trắng ở hai cạnh để giữ đúng tỷ lệ gốc trước khi đưa vào mạng."*

#### Câu 5: Em đã làm sạch dữ liệu trong file `Model/preprocess.py` như thế nào?
* **Ý đồ Giám khảo**: Kiểm tra xem bạn có thực sự đọc và hiểu code trong `Model/preprocess.py` không.
* **Đáp án điểm 10**:
  > *"Trong file `preprocess.py`, em thực hiện 3 bước làm sạch cốt lõi:  
  > 1. Kiểm tra sự tồn tại thực tế của từng tệp ảnh trên ổ đĩa thông qua hàm `os.path.exists()`, loại bỏ các dòng metadata không có ảnh kèm theo.  
  > 2. Điền giá trị mặc định cho các trường dữ liệu bị thiếu (ví dụ `subCategory` bị null thì điền 'Other', `usage` bị null thì điền 'Casual').  
  > 3. Định nghĩa lớp `FashionDataset` kế thừa `torch.utils.data.Dataset` và chia tập dữ liệu Train 90% - Val 10% để kiểm định tính ổn định."*

#### Câu 6: Tại sao em lại chia tập dữ liệu 90% Train và 10% Validation mà không dùng 80/20 hay K-Fold?
* **Ý đồ Giám khảo**: Kiểm tra tư duy thiết kế thực nghiệm.
* **Đáp án điểm 10**:
  > *"Thưa Thầy Cô, vì ở đây em sử dụng mô hình Transfer Learning với trọng số đã đóng băng cố định (Feature Extraction), tập dữ liệu không dùng để cập nhật trọng số gradient mà dùng để xây dựng kho vector đặc trưng (Feature Store).  
  > Em dành 90% (1,080 mẫu) để làm phong phú kho sản phẩm đề xuất, và trích 10% (120 mẫu) độc lập để kiểm định chất lượng phân bố vector và độ trễ của pipeline. Tỷ lệ 90/10 giúp tối đa hóa số lượng trang phục có sẵn trong kho gợi ý."*

---

## Chuyên đề 2: Kiến trúc Deep Learning & ResNet50

#### Câu 7: Tại sao em lại chọn ResNet50 mà không dùng VGG16, MobileNetV2 hay Vision Transformer (ViT)?
* **Ý đồ Giám khảo**: Kiểm tra khả năng so sánh kiến trúc và lý do lựa chọn giải pháp.
* **Đáp án điểm 10**:
  > *"Thưa Thầy Cô, em đã cân nhắc kỹ giữa các kiến trúc:  
  > 1. **So với VGG16**: VGG16 quá nặng (hơn 138 triệu tham số, dung lượng file hơn 500MB), tốn RAM và chạy rất chậm trên CPU. ResNet50 chỉ có 25 triệu tham số nhờ thiết kế Bottleneck và tầng GAP.  
  > 2. **So với MobileNetV2**: MobileNet rất nhẹ nhưng khả năng trích xuất đặc trưng thời trang tinh vi (chất liệu vải, đường may) không sâu bằng ResNet50.  
  > 3. **So với Vision Transformer (ViT)**: ViT đòi hỏi tập dữ liệu huấn luyện khổng lồ và phần cứng GPU mạnh để suy luận, không phù hợp với mục tiêu triển khai nhẹ nhàng trên CPU trong 48 giờ.  
  > Do đó, **ResNet50 là điểm cân bằng hoàn hảo nhất** giữa độ chính xác thị giác và tốc độ suy luận thời gian thực."*

#### Câu 8: Transfer Learning ở đây hoạt động như thế nào? Em có fine-tune không hay chỉ Feature Extraction?
* **Ý đồ Giám khảo**: Bẫy phân biệt giữa Feature Extraction và Fine-Tuning.
* **Đáp án điểm 10**:
  > *"Dạ thưa Thầy Cô, trong đồ án này em áp dụng chiến lược **Feature Extraction (Trích xuất đặc trưng cố định)**. Em đóng băng toàn bộ các tầng tích chập của mạng ResNet50 bằng lệnh `param.requires_grad = False` và loại bỏ tầng Fully Connected 1,000 lớp.  
  > Lý do em không fine-tune là vì tập dữ liệu 1,200 ảnh là tương đối nhỏ. Nếu mở mở khóa trọng số để fine-tune mà không có nhãn giám sát cặp outfit chuẩn, mạng rất dễ bị Overfitting. Khả năng nhận diện đường nét, màu sắc, hình khối học từ 1.28 triệu ảnh ImageNet đã quá đủ xuất sắc để biểu diễn không gian thời trang."*

#### Câu 9: Trình bày chứng minh toán học của Skip Connection trong ResNet giải quyết hiện tượng Triệt tiêu Đạo hàm?
* **Ý đồ Giám khảo**: Kiểm tra bản chất toán học sâu của Deep Learning (Câu hỏi phân loại điểm 9-10).
* **Đáp án điểm 10**:
  > *"Thưa Thầy Cô, ở mạng nơ-ron thông thường, đầu ra là $y = \mathcal{F}(x)$. Khi lan truyền ngược qua 50 tầng, theo quy tắc chuỗi (Chain Rule), gradient là tích của 50 ma trận trọng số. Nếu các trọng số nhỏ hơn 1, gradient sẽ suy giảm về 0 (Vanishing Gradient).  
  > ResNet bổ sung đường truyền tắt (Identity Shortcut), biến đầu ra thành:  
  > $$y = \mathcal{F}(x) + x$$  
  > Khi tính đạo hàm hàm mất mát $\mathcal{L}$ theo đầu vào $x$:  
  > $$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left( \frac{\partial \mathcal{F}}{\partial x} + 1 \right) = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial \mathcal{F}}{\partial x} + \frac{\partial \mathcal{L}}{\partial y}$$  
  > **Ý nghĩa mấu chốt nằm ở số $+1$**: Dù đạo hàm $\frac{\partial \mathcal{F}}{\partial x}$ có bị triệt tiêu về $0$, thành phần đạo hàm $\frac{\partial \mathcal{L}}{\partial y}$ vẫn luôn được truyền trực tiếp về các tầng trước mà không bao giờ bị mất đi!"*

#### Câu 10: Tầng Global Average Pooling (GAP) là gì? Khác gì so với Flatten và Fully Connected truyền thống?
* **Ý đồ Giám khảo**: Kiểm tra sự hiểu biết về cách trích xuất vector 2048 chiều.
* **Đáp án điểm 10**:
  > *"Dạ thưa Thầy Cô, ở mạng CNN cổ điển như VGG, tầng cuối cùng dùng **Flatten** để kéo phẳng ma trận $7 \times 7 \times 512$ thành vector 25,088 chiều, sau đó nối vào tầng Fully Connected. Cách làm này sinh ra hàng chục triệu tham số và rất dễ bị Overfitting.  
  > Ngược lại, **Global Average Pooling (GAP)** lấy trung bình cộng của toàn bộ điểm ảnh trên mỗi kênh bản đồ đặc trưng:  
  > $$\text{GAP}(c) = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W X_{i, j, c}$$  
  > Bản đồ đặc trưng $7 \times 7 \times 2048$ được nén thành đúng **1 vector 2048 chiều**. GAP không sinh thêm bất kỳ tham số nào, bảo toàn thông tin không gian toàn cục và giúp mô hình cực kỳ nhẹ."*

#### Câu 11: Vector 2048 chiều có quá lớn không? Tại sao em không dùng PCA để giảm chiều xuống 128?
* **Ý đồ Giám khảo**: Kiểm tra tư duy tối ưu hóa hiệu năng và cân nhắc trade-off.
* **Đáp án điểm 10**:
  > *"Thưa Thầy Cô, việc dùng PCA giảm chiều là một giải pháp tốt khi kho dữ liệu lên tới hàng chục triệu sản phẩm.  
  > Tuy nhiên, với quy mô 1,200 sản phẩm trong đồ án này, ma trận $1200 \times 2048$ chỉ chiếm đúng **9.83 Megabytes RAM**. Thời gian tính Dot Product trên ma trận này chỉ mất **0.635 mili-giây** trên CPU.  
  > Giữ nguyên 2048 chiều giúp bảo toàn trọn vẹn các chi tiết vi mô của trang phục (như cúc áo, nẹp cổ, chất liệu vải) mà không bị mất mát thông tin qua phép chiếu tuyến tính của PCA."*

#### Câu 12: Làm thế nào em tối ưu hóa tốc độ trích xuất đặc trưng trong file `Model/train_and_val.py`?
* **Ý đồ Giám khảo**: Kiểm tra kỹ năng lập trình PyTorch thực chiến.
* **Đáp án điểm 10**:
  > *"Trong file `Model/train_and_val.py`, em tối ưu hóa qua 3 kỹ thuật:  
  > 1. Sử dụng `DataLoader` với kỹ thuật **Batch Processing (batch_size = 32)** để đẩy đồng thời 32 ảnh qua mạng, tận dụng khả năng tính toán ma trận song song của CPU/GPU, nhanh hơn gấp 5 lần so với chạy từng ảnh lẻ.  
  > 2. Đặt `torch.no_grad()` để tắt hoàn toàn đồ thị tính đạo hàm ngược, tiết kiệm 60% dung lượng RAM.  
  > 3. Sau khi trích xuất, em chuẩn hóa $L_2$-norm sẵn và lưu ra file nhị phân `embeddings.pkl` để Web App nạp thẳng vào bộ nhớ mà không cần tính toán lại."*

---

## Chuyên đề 3: Toán học, Embeddings & Cosine Similarity

#### Câu 13: Tại sao em dùng Cosine Similarity thay vì Khoảng cách Euclid ($L_2$ Distance)?
* **Ý đồ Giám khảo**: Đây là câu hỏi kinh điển nhất của mọi hội đồng chấm Recommender System.
* **Đáp án điểm 10**:
  > *"Thưa Thầy Cô, có 2 lý do toán học và thực tiễn cốt lõi:  
  > 1. **Về mặt quang học**: Độ dài của vector phụ thuộc vào độ tương phản và ánh sáng ảnh chụp. Hai bức ảnh chụp cùng một chiếc áo ở điều kiện sáng mạnh và yếu sẽ có khoảng cách Euclid rất xa nhau, nhưng góc phương vị $\theta$ giữa hai vector là bằng $0^\circ$. Cosine Similarity chỉ đo góc $\theta$, giúp triệt tiêu hoàn toàn sự sai khác về ánh sáng.  
  > 2. **Về mặt số chiều**: Trong không gian 2048 chiều, khoảng cách Euclid gặp hiện tượng tập trung khoảng cách (Distance Concentration), khoảng cách giữa mọi cặp điểm đều xấp xỉ như nhau. Cosine Similarity phản ánh tỷ lệ đóng góp của từng thuộc tính thị giác, cho kết quả tương đồng phong cách chính xác hơn nhiều."*

#### Câu 14: Hiện tượng "Bẫy số chiều" (Curse of Dimensionality) là gì?
* **Ý đồ Giám khảo**: Kiểm tra kiến thức đại số tuyến tính và hình học không gian nhiều chiều.
* **Đáp án điểm 10**:
  > *"Dạ thưa Thầy Cô, 'Bẫy số chiều' xảy ra khi số chiều không gian $D$ tăng lên rất lớn (ở đây là $D = 2048$). Khi đó, thể tích của không gian tăng theo hàm mũ, khiến dữ liệu trở nên cực kỳ thưa thớt (Sparse).  
  > Tỷ số giữa khoảng cách tới điểm xa nhất và điểm gần nhất tiệm cận về 1:  
  > $$\lim_{D \to \infty} \frac{\text{dist}_{\max} - \text{dist}_{\min}}{\text{dist}_{\min}} \to 0$$  
  > Điều này làm cho các độ đo khoảng cách tuyệt đối như Euclid bị vô hiệu hóa vì điểm nào trông cũng xa như nhau. Đo góc cosin là giải pháp chuẩn công nghiệp để khắc phục hiện tượng này."*

#### Câu 15: Chuẩn hóa $L_2$-Norm biến phép tính Cosine Similarity thành phép tính gì?
* **Ý đồ Giám khảo**: Kiểm tra sự thấu hiểu về tối ưu hóa tính toán ma trận.
* **Đáp án điểm 10**:
  > *"Thưa Thầy Cô, công thức Cosine tổng quát là:  
  > $$\text{Cosine}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$  
  > Khi cả hai vector đã được chuẩn hóa $L_2$-Norm ($\|u\|_2 = 1.0$ và $\|v\|_2 = 1.0$), mẫu số trở thành $1 \times 1 = 1$.  
  > Khi đó, Cosine Similarity **chính là Tích vô hướng (Dot Product)**:  
  > $$\text{Cosine}(u, v) = u \cdot v = \sum_{i=1}^D u_i v_i$$  
  > Điều này biến việc so sánh 1 sản phẩm với toàn bộ 1,200 sản phẩm trong kho thành **1 phép nhân ma trận đơn lẻ**: `np.dot(matrix, vec.T)`. Nhờ vậy, thời gian truy vấn chỉ mất 0.6 mili-giây!"*

#### Câu 16: Hàm mục tiêu tính điểm Outfit Compatibility Score của em có công thức thế nào? Các hệ số 0.80, 0.12, 0.20 có ý nghĩa gì?
* **Ý đồ Giám khảo**: Kiểm tra tính minh bạch và logic của thuật toán phối đồ.
* **Đáp án điểm 10**:
  > *"Hàm mục tiêu tính điểm hòa hợp của em là:  
  > $$\text{Score}(q, c) = 0.80 \cdot \text{Cosine}(v_q, v_c) + 0.12 \cdot \mathbb{I}_{(\text{cùng hoàn cảnh})} + 0.20$$  
  > *Ý nghĩa của từng thành phần*:  
  > - $0.80 \cdot \text{Cosine}(v_q, v_c)$: Chiếm 80% trọng số, đảm bảo hai món đồ phải có sự đồng điệu về phong cách thị giác, form dáng và chất liệu từ ResNet50.  
  > - $0.12 \cdot \mathbb{I}_{(\text{cùng hoàn cảnh})}$: Điểm thưởng nếu cả hai món đều thuộc cùng hoàn cảnh sử dụng (ví dụ cùng là đồ Formal đi làm hoặc cùng là Casual đi chơi).  
  > - $+0.20$: Hệ số dịch chuyển cơ bản (Base Bias) để co giãn điểm số về khoảng hiển thị trực quan $[50\%, 99\%]$ trên giao diện Web, giúp người dùng dễ hiểu điểm phần trăm."*

#### Câu 17: Quy tắc phối đồ bù trừ (Complementary Pairing Logic) được em cài đặt như thế nào?
* **Ý đồ Giám khảo**: Kiểm tra xem bạn có cài đặt luật hay để mô hình gợi ý bừa bãi.
* **Đáp án điểm 10**:
  > *"Em xây dựng hàm `get_target_pairing_categories()` với quy tắc bù trừ danh mục:  
  > - Nếu đầu vào là **Topwear (Áo)** hoặc **Dress**: Hệ thống bắt buộc phải tìm các ứng viên thuộc nhóm **Bottomwear (Quần/Váy)**, **Shoes (Giày dép)**, và **Watches/Bags (Phụ kiện)**.  
  > - Nếu đầu vào là **Bottomwear (Quần)**: Hệ thống quay ngược lại tìm **Topwear (Áo)** và Giày dép.  
  > - Ngoài ra, hệ thống tự động lọc theo điều kiện giới tính: Đồ Nam chỉ phối cùng đồ Nam hoặc Unisex, đảm bảo không có trường hợp gợi ý áo nam phối với váy nữ."*

#### Câu 18: Nếu món đồ đầu vào là Phụ kiện (Đồng hồ hoặc Kính mắt) thì hệ thống phối đồ ra sao?
* **Ý đồ Giám khảo**: Kiểm tra trường hợp biên (Edge Case).
* **Đáp án điểm 10**:
  > *"Thưa Thầy Cô, khi người dùng đưa vào một phụ kiện (ví dụ chiếc đồng hồ thể thao G-Shock), quy tắc bù trừ của em sẽ thiết lập mục tiêu tìm kiếm cả Áo (Topwear), Quần (Bottomwear) và Giày (Shoes).  
  > Hệ thống sẽ trích xuất vector phong cách của chiếc đồng hồ (mạnh mẽ, thể thao, năng động) và truy vấn tìm ra chiếc áo thun và đôi giày sneaker có phong cách thể thao tương đồng nhất."*

---

## Chuyên đề 4: Thực nghiệm, Kiểm thử & Đánh giá Đối chứng

#### Câu 19: Bài toán này không có nhãn đúng/sai (Ground Truth Outfit), em đánh giá chất lượng bằng cách nào?
* **Ý đồ Giám khảo**: Câu hỏi hóc búa về phương pháp luận nghiên cứu (Evaluation Methodology).
* **Đáp án điểm 10**:
  > *"Thưa Thầy Cô, đây chính là đặc thù của bài toán Khám phá Thời trang (Fashion Recommendation) vì cái đẹp và phong cách ăn mặc mang tính thẩm mỹ chủ quan (Subjective).  
  > Để đánh giá khoa học, em kết hợp 2 phương pháp:  
  > 1. **Đánh giá Định lượng (Quantitative Benchmark)**: Đo lường tốc độ suy luận (Latency < 1ms/query), dung lượng bộ nhớ (Memory footprint < 10MB), và kiểm tra độ ổn định toán học (không có giá trị NaN/Inf, vector chuẩn hóa đạt độ dài đúng bằng 1.0).  
  > 2. **Đánh giá Định tính (Qualitative Visual Evaluation)**: Xây dựng các Test Cases đối chứng trực diện giữa mô hình Deep Learning ResNet50 và mô hình Baseline Color Histogram để các chuyên gia và người dùng mắt thường thẩm định tính thẩm mỹ của set đồ được phối."*

#### Câu 20: Mô hình Baseline Color Histogram hoạt động ra sao và nó thua kém ResNet50 ở điểm nào?
* **Ý đồ Giám khảo**: Kiểm tra xem bạn có hiểu mô hình Baseline tự tạo không.
* **Đáp án điểm 10**:
  > *"Mô hình Baseline chuyển ảnh sang không gian màu HSV, chia 8 bins cho mỗi kênh Hue, Saturation, Value $\rightarrow 8 \times 8 \times 8 = 512$ chiều.  
  > Điểm yếu cốt tử của Baseline là nó **hoàn toàn mù tịt về mặt ngữ nghĩa hình học**: Nó chỉ đếm xem có bao nhiêu pixel màu xám hay màu đỏ. Do đó, một chiếc áo sơ mi công sở kẻ sọc và một chiếc quần đùi ngủ kẻ sọc nếu có cùng tỷ lệ màu thì Baseline xem là giống hệt nhau.  
  > ResNet50 vượt trội vì nó nhận biết được ve áo sơ mi, cạp quần âu, form áo vest, từ đó gợi ý set đồ chuẩn mực về hoàn cảnh sử dụng."*

#### Câu 21: File `Model/test_model.py` kiểm thử những nội dung gì?
* **Ý đồ Giám khảo**: Kiểm tra xem bạn có nắm cấu trúc kiểm thử trong thư mục `Model/` không.
* **Đáp án điểm 10**:
  > *"File `Model/test_model.py` thực thi 3 bài test độc lập:  
  > - **Bài test 1 (Định lượng)**: Chạy vòng lặp 50 truy vấn đo thời gian Cosine Similarity trung bình, đạt 0.635 ms/sản phẩm (~1574 QPS).  
  > - **Bài test 2 (Định tính)**: Chạy 2 Test Cases thực tế (Nam công sở và Nữ dạo phố) và tự động xuất ảnh set đồ vào thư mục `Model/test_results/`.  
  > - **Bài test 3 (Độ bền)**: Đưa vào một vector nhiễu ngẫu nhiên để kiểm tra xem hệ thống có bị crash hay tràn số NaN không. Kết quả đạt Pass 100%."*

#### Câu 22: Tốc độ suy luận của toàn bộ hệ thống là bao nhiêu? Có đạt chuẩn Real-time không?
* **Ý đồ Giám khảo**: Kiểm tra tính khả thi khi vận hành thực tế.
* **Đáp án điểm 10**:
  > *"Thưa Thầy Cô, hệ thống đạt chuẩn Real-time tuyệt đối:  
  > - Thời gian tìm kiếm Cosine trên toàn kho 1,200 sản phẩm: **0.635 mili-giây**.  
  > - Thời gian tiền xử lý ảnh người dùng tải lên và trích xuất vector qua ResNet50: **~28 mili-giây** trên CPU phổ thông.  
  > Tổng thời gian từ lúc người dùng click chọn đến khi giao diện Web Streamlit hiển thị trọn bộ outfit chỉ khoảng **30 mili-giây**, nhanh hơn nhiều so với tiêu chuẩn phản hồi web 200 mili-giây thông thường."*

#### Câu 23: Nếu người dùng upload một bức ảnh không phải quần áo (ví dụ ảnh con mèo), hệ thống sẽ xử lý thế nào?
* **Ý đồ Giám khảo**: Kiểm tra khả năng xử lý ngoại lệ thực tế (Out-of-Distribution Input).
* **Đáp án điểm 10**:
  > *"Hiện tại, ResNet50 vẫn sẽ trích xuất vector đặc trưng của bức ảnh đó và tìm sản phẩm thời trang có màu sắc, hình khối gần nhất trong kho để nhận diện thuộc tính trước khi phối đồ.  
  > Để hoàn thiện trong bản Production, em sẽ bổ sung một **Tầng Phân loại Lọc trước (Out-of-Domain Classifier)** ở cổng đầu vào: Kiểm tra xem ảnh tải lên có thuộc danh mục thời trang không; nếu độ tin cậy dưới 50% sẽ thông báo cho người dùng yêu cầu chụp lại trang phục."*

#### Câu 24: Kết quả cụ thể của Test Case 1 (Nam công sở) trong file test của em là gì?
* **Ý đồ Giám khảo**: Kiểm tra xem bạn có trực tiếp xem kết quả test không.
* **Đáp án điểm 10**:
  > *"Ở Test Case 1: Món gốc là Áo sơ mi xanh dương nam Mark Taylor Men Plain Blue Shirts.  
  > Hệ thống gợi ý:  
  > 1. Quần âu nâu sẫm Peter England Trouser với điểm tương thích **84.4%**.  
  > 2. Giày da công sở / Sneaker da thanh lịch đạt **50.0%**.  
  > 3. Đồng hồ nam Casio G-Shock lịch lãm đạt **50.0%**.  
  > Ảnh kết quả được hệ thống tự động xuất ra và lưu trữ tại `Model/test_results/test_case_men_formal.png`."*

---

## Chuyên đề 5: Hệ thống, Mở rộng Production & Thực tế

#### Câu 25: Nếu kho sản phẩm tăng từ 1,200 lên 1,000,000 sản phẩm, thuật toán tìm kiếm tuyến tính hiện tại có bị chậm không? Giải pháp là gì?
* **Ý đồ Giám khảo**: Kiểm tra tư duy kiến trúc hệ thống quy mô lớn (Scalability).
* **Đáp án điểm 10**:
  > *"Thưa Thầy Cô, hiện tại với 1,200 sản phẩm, thuật toán tìm kiếm tuyến tính (Exhaustive Linear Scan) chỉ mất 0.6ms. Nhưng khi kho dữ liệu tăng lên 1,000,000 sản phẩm, độ phức tạp $O(N)$ sẽ khiến thời gian tính toán tăng lên khoảng 500ms - 1 giây, gây trễ giao diện.  
  > Giải pháp tối ưu công nghiệp là chuyển từ Linear Search sang **Tìm kiếm Láng giềng Gần đúng (Approximate Nearest Neighbors - ANN)** bằng thư viện **FAISS của Meta** hoặc **Annoy của Spotify**.  
  > Bằng cách phân cụm không gian vector với thuật toán Inverted File Index (IVF) và lượng tử hóa tích phân (Product Quantization), thời gian tìm kiếm trên 1 triệu sản phẩm sẽ giảm xuống dưới **3 mili-giây** mà chỉ hy sinh dưới 1% độ chính xác."*

#### Câu 26: Vấn đề Cold Start (khi có một chiếc áo hoàn toàn mới nhập vào kho) được giải quyết ra sao?
* **Ý đồ Giám khảo**: Kiểm tra kiến thức về các vấn đề kinh điển của Recommender Systems.
* **Đáp án điểm 10**:
  > *"Đây chính là ưu điểm vượt trội của phương pháp **Content-Based Feature Extraction bằng Deep Learning** so với phương pháp Lọc cộng tác (Collaborative Filtering).  
  > Trong Collaborative Filtering, sản phẩm mới chưa có ai click hay mua thì không thể gợi ý (Cold Start).  
  > Nhưng trong hệ thống của em, sản phẩm mới chỉ cần có 1 bức ảnh chụp: Ngay lập tức mạng ResNet50 trích xuất được vector đặc trưng trong 28ms và đưa vào kho dữ liệu. Sản phẩm mới có thể được phối đồ và gợi ý cho khách hàng ngay giây tiếp theo mà không cần chờ lịch sử tương tác."*

#### Câu 27: Tại sao em lại tách dự án thành thư mục `Model/` và `app_web/`?
* **Ý đồ Giám khảo**: Kiểm tra tư duy thiết kế phần mềm và tính trung thực học thuật.
* **Đáp án điểm 10**:
  > *"Dạ thưa Thầy Cô, em tách bạch dự án theo đúng nguyên lý **Phân tách Trách nhiệm (Separation of Concerns)**:  
  > - Thư mục **`Model/`** là **lõi khoa học của đồ án**: Chứa toàn bộ các mô-đun tiền xử lý dữ liệu, kiến trúc ResNet50, và kịch bản kiểm thử độc lập mà em tự lực nghiên cứu và làm chủ.  
  > - Thư mục **`app_web/`** là **tầng giao diện ứng dụng người dùng**: Được em xây dựng với sự trợ giúp của AI để tạo giao diện Streamlit tương tác nhanh chóng, phục vụ việc trình chiếu trực quan cho buổi bảo vệ hôm nay mà không làm loãng phần nghiên cứu thuật toán cốt lõi."*

#### Câu 28: Nếu đưa vào thực tế trên sàn Shopee hay Tiki, hệ thống này sẽ tích hợp theo kiến trúc nào?
* **Ý đồ Giám khảo**: Kiểm tra tầm nhìn triển khai Production.
* **Đáp án điểm 10**:
  > *"Hệ thống sẽ được triển khai theo kiến trúc **Microservices**:  
  > 1. **Offline Pipeline (Batch)**: Khi nhà bán hàng đăng tải sản phẩm mới, worker chạy ngầm sẽ gọi `train_and_val.py` để trích xuất vector và đẩy vào cơ sở dữ liệu vector như Milvus hoặc Pinecone.  
  > 2. **Online Inference (Real-time)**: Giao diện web/mobile app của sàn TMĐT gọi qua API RESTful (FastAPI). Khi người dùng mở trang chi tiết sản phẩm, API chỉ cần truy vấn vector đã trích xuất sẵn từ Vector DB và trả về danh sách outfit gợi ý trong vòng 5ms."*

#### Câu 29: Nếu khách hàng là người theo phong cách tối giản (Minimalism) hoặc phong cách đường phố (Streetwear), mô hình có nhận diện được không?
* **Ý đồ Giám khảo**: Kiểm tra khả năng cá nhân hóa phong cách sâu.
* **Đáp án điểm 10**:
  > *"Mạng ResNet50 nhận diện được các đặc trưng hình thái này thông qua các tầng tích chập sâu: Áo phông Streetwear thường có form rộng (Oversized), họa tiết đồ họa lớn ở ngực, màu sắc tương phản; trong khi đồ Minimalism có bề mặt vải trơn, màu đơn sắc trung tính, không họa tiết.  
  > Nhờ vector 2048 chiều phân tách rõ ràng các nhóm phong cách này trong không gian hình học, thuật toán sẽ tự động gán điểm tương đồng cao cho các món đồ cùng phong cách."*

#### Câu 30: Trong tương lai, em có kế hoạch phát triển Virtual Try-on (Thử đồ ảo) như thế nào?
* **Ý đồ Giám khảo**: Kiểm tra định hướng phát triển công nghệ Generative AI.
* **Đáp án điểm 10**:
  > *"Thưa Thầy Cô, sau khi AI đã gợi ý được bộ outfit đẹp mắt, bước đột phá tiếp theo là giúp khách hàng nhìn thấy chính mình mặc bộ đồ đó trước khi bấm nút 'Mua ngay'.  
  > Em sẽ kết hợp 2 công nghệ:  
  > 1. **YOLOv8-Segmentation**: Bóc tách chính xác phần thân trên, thân dưới và đôi chân từ ảnh chụp người thật của khách hàng.  
  > 2. **Generative Diffusion Model (như OOTDiffusion hoặc IDM-VTON)**: Điều chỉnh nếp gấp vải, độ rủ và ánh sáng để ghép trang phục được gợi ý vào cơ thể người dùng một cách chân thực nhất. Đây sẽ là bước hoàn thiện trọn vẹn chu trình mua sắm thời trang thông minh."*

---

# PHẦN 3: BẢNG TỔNG HỢP CÁC CON SỐ "VÀNG" CẦN NHỚ KHI ĐỨNG BẢO VỆ

| Thông số kỹ thuật | Con số cần nói ngay | Ý nghĩa chứng minh |
| :--- | :--- | :--- |
| **Quy mô tập dữ liệu** | **1,200 sản phẩm thật** | Dữ liệu thật từ sàn Myntra (Kaggle), 100% ảnh sạch. |
| **Kích thước ảnh chuẩn hóa** | **224 &times; 224 pixels** | Kích thước hình học đầu vào chuẩn của mạng CNN ImageNet. |
| **Phân phối chuẩn hóa** | **Mean = [0.485, 0.456, 0.406]**<br>**Std = [0.229, 0.224, 0.225]** | Chuẩn hóa Z-Score ImageNet khớp phân phối điểm ảnh. |
| **Kiến trúc mô hình chính** | **ResNet50 Transfer Learning** | 50 tầng tích chập, đóng băng trọng số, khai thác tầng GAP. |
| **Số chiều vector ResNet50** | **2048 chiều** | Tọa độ biểu diễn ngữ nghĩa thời trang sâu. |
| **Số chiều vector Baseline** | **512 chiều** | Phân bố biểu đồ màu HSV 3D ($8\times 8\times 8$). |
| **Thời gian truy vấn kho** | **0.635 mili-giây / query** | Tốc độ tìm kiếm ma trận siêu tốc (~1574 truy vấn/giây). |
| **Thời gian suy luận toàn bộ** | **&lt; 30 mili-giây** | Đạt chuẩn Real-time trên CPU cá nhân thông thường. |
| **Dung lượng ma trận kho** | **9.83 MB** | Vô cùng nhẹ, không tốn tài nguyên phần cứng. |
| **Tỷ lệ phân chia tập mẫu** | **90% Train / 10% Validation** | Tối đa hóa số lượng mẫu gợi ý trong kho sản phẩm. |
