# Hướng dẫn sử dụng Hệ thống EVIS (Enhanced Visual Information Search)

Tài liệu này hướng dẫn chi tiết cách lập trình và sử dụng các module cốt lõi của hệ thống EVIS. Thay vì chỉ chạy các script có sẵn, bạn có thể tích hợp các thành phần này vào quy trình xử lý dữ liệu hoặc ứng dụng của riêng mình.

## 1. Khởi tạo các thành phần trích xuất đặc trưng

### CLIPExtractor
Dùng để trích xuất đặc trưng hình ảnh và văn bản vào cùng một không gian vector (512 chiều).

```python
from src.features.clip_extractor import CLIPExtractor

# Khởi tạo (tự động chọn CUDA nếu có)
extractor = CLIPExtractor(model_name="openai/clip-vit-base-patch16")

# Trích xuất đặc trưng ảnh
image_features = extractor.extract_image_features("path/to/image.jpg") # Trả về torch.Tensor (1, 512)

# Trích xuất đặc trưng văn bản
text_features = extractor.extract_text_features("một người đang lái xe") # Trả về torch.Tensor (1, 512)
```

### OCRExtractor
Dùng để nhận diện chữ viết trong ảnh và chuyển đổi chúng thành vector đặc trưng.

```python
from src.features.ocr_extractor import OCRExtractor

# Khởi tạo với ngôn ngữ hỗ trợ (mặc định là tiếng Anh 'en')
ocr_module = OCRExtractor(languages=['en'], gpu=True)

# Trích xuất văn bản từ ảnh
text, details = ocr_module.extract_text("path/to/image.jpg")
# Trả về:
# - text: Chuỗi văn bản thô (string)
# - details: Danh sách các dict chứa thông tin chi tiết (bbox, confidence)

# Chuyển đổi văn bản OCR thành embedding (cùng không gian với CLIP)
ocr_embedding = ocr_module.get_ocr_embedding(text) # Trả về torch.Tensor (1, 512)
```

## 2. Xử lý đặc trưng và Metadata

### EventContextEncoder
Mã hóa thông tin ngữ cảnh như thời gian, địa điểm và tên sự kiện thành một vector 512 chiều.

```python
from src.models.event_encoder import EventContextEncoder
import torch

encoder = EventContextEncoder()

# Giả sử chúng ta có vector tên sự kiện từ CLIP
event_name_emb = extractor.extract_text_features("Sự kiện giao thông")

# Mã hóa ngữ cảnh (Thời gian, Vĩ độ, Kinh độ)
# timestamp có thể là định dạng ISO hoặc unix timestamp
context_emb = encoder(
    event_name_emb, 
    timestamp="2023-10-27T10:30:00Z", 
    lat=10.762622, 
    lon=106.660172
) # Trả về torch.Tensor (1, 512)
```

## 3. Sử dụng Fusion Module

### MultimodalFusion
Kết hợp các loại đặc trưng khác nhau (Hình ảnh, OCR, Ngữ cảnh) bằng cơ chế Cross-Attention.

```python
from src.models.fusion_module import MultimodalFusion

# Khởi tạo module fusion sử dụng cross-attention
fusion_module = MultimodalFusion(fusion_type="cross_attention")

# Kết hợp các đặc trưng
# Các tensor đầu vào đều có kích thước (batch_size, 512)
unified_embedding = fusion_module(
    visual_emb=image_features, 
    ocr_emb=ocr_embedding, 
    event_emb=context_emb
) # Trả về torch.Tensor (batch_size, 512)
```

## 4. Quản lý Index và Tìm kiếm

### VectorIndex
Lưu trữ và thực hiện tìm kiếm vector hiệu năng cao bằng FAISS.

```python
from src.indexing.vector_db import VectorIndex
import numpy as np

# Khởi tạo index
index = VectorIndex(dimension=512, metric="cosine")

# Thêm dữ liệu vào index
vectors = np.random.randn(10, 512).astype('float32') # Giả lập vector đặc trưng
metadata = [{"image_path": f"img_{i}.jpg", "event": "traffic"} for i in range(10)]
index.add_vectors(vectors, metadata)

# Lưu index ra file
index.save("my_index")

# Tải index từ file
index.load("my_index")
```

### SearchEngine
Lớp cấp cao hỗ trợ tìm kiếm từ câu truy vấn văn bản.

```python
from src.retrieval.search_engine import SearchEngine

# Khởi tạo engine với index đã có
engine = SearchEngine(vector_db_path="my_index")

# Thực hiện tìm kiếm
results = engine.search(query="xe máy trên đường phố", top_k=5)
# Trả về: Danh sách metadata của các kết quả gần nhất kèm theo 'score'
```

## 5. Giải thích kết quả (XAI)

### AttentionVisualizer
Tạo bản đồ nhiệt (heatmap) để giải thích lý do tại sao mô hình lại chọn hình ảnh đó cho câu truy vấn cụ thể.

```python
from src.visualization.explainability import AttentionVisualizer

visualizer = AttentionVisualizer(extractor=extractor)

# Tạo heatmap
heatmap = visualizer.generate_heatmap("path/to/image.jpg", "xe máy")

# Chồng heatmap lên ảnh gốc
overlaid_image = visualizer.overlay_heatmap("path/to/image.jpg", heatmap)

# Lưu kết quả
visualizer.save_visualization(overlaid_image, "explanation_result.jpg")
```

## 6. Ví dụ Code hoàn chỉnh: Chu kỳ tìm kiếm từ đầu đến cuối

```python
import torch
from src.features.clip_extractor import CLIPExtractor
from src.retrieval.search_engine import SearchEngine

def full_pipeline_demo():
    # 1. Khởi tạo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = SearchEngine(vector_db_path="data/indices/base_index", device=device)
    
    # 2. Thực hiện tìm kiếm từ câu truy vấn của người dùng
    query = "biển quảng cáo màu đỏ trên phố"
    print(f"Đang tìm kiếm cho: '{query}'...")
    
    results = engine.search(query, top_k=3)
    
    # 3. Hiển thị kết quả và giải thích kết quả tốt nhất
    if results:
        best_match = results[0]
        print(f"Kết quả tốt nhất: {best_match['image_path']} (Score: {best_match['score']:.4f})")
        
        # Tạo giải thích XAI
        explain_path = engine.explain_result(best_match, query, output_dir="results/explanations")
        print(f"Ảnh giải thích đã được lưu tại: {explain_path}")
    else:
        print("Không tìm thấy kết quả phù hợp.")

if __name__ == "__main__":
    full_pipeline_demo()