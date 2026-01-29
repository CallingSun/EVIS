# EVIS: Hệ thống Tìm kiếm Thông tin Hình ảnh có tính Giải thích (Explainable Visual Information System)

EVIS là một hệ thống tìm kiếm đa phương thức tiên tiến, cho phép kết hợp Hình ảnh, Văn bản, OCR (Nhận diện chữ viết trong ảnh) và Metadata của sự kiện (thời gian, địa điểm, tên sự kiện) vào một không gian vector thống nhất. Hệ thống sử dụng các mô hình State-of-the-Art như CLIP để trích xuất đặc trưng và tích hợp khả năng giải thích (Explainability) thông qua bản đồ nhiệt (Heatmaps).

## 🚀 Tính năng chính

- **Hợp nhất Đa phương thức (Multimodal Fusion)**: Kết hợp đặc trưng hình ảnh (CLIP), văn bản OCR (EasyOCR/PaddleOCR) và ngữ cảnh sự kiện (thời gian, tọa độ, tên sự kiện) bằng cơ chế Cross-Attention.
- **Trí tuệ Nhân tạo có tính Giải thích (XAI)**: Tạo bản đồ nhiệt (Heatmaps) để chỉ ra các vùng trong ảnh mà mô hình tập trung vào khi thực hiện một truy vấn cụ thể.
- **Tìm kiếm Thống nhất**: Thực hiện các truy vấn phức tạp kết hợp nội dung hình ảnh, chữ viết xuất hiện trong ảnh và ngữ cảnh của sự kiện.
- **Lập chỉ mục Hiệu quả**: Sử dụng thư viện FAISS để tìm kiếm tương đồng nhanh chóng trên quy mô dữ liệu lớn.

## 🛠️ Yêu cầu hệ thống và Cài đặt

Để chạy hệ thống, bạn cần cài đặt các thư viện sau:

```bash
pip install torch torchvision torchaudio
pip install transformers pillow easyocr numpy opencv-python faiss-cpu
```

*Lưu ý: Để tăng tốc độ xử lý bằng GPU, hãy đảm bảo bạn đã cài đặt phiên bản PyTorch phù hợp với CUDA.*

## 📂 Cấu trúc thư mục dự án

```text
.
├── src/                    # Logic cốt lõi của hệ thống
│   ├── features/           # Trích xuất đặc trưng CLIP và OCR
│   ├── models/             # Mô hình Fusion và Encoder cho ngữ cảnh sự kiện
│   ├── indexing/           # Quản lý Pipeline và Cơ sở dữ liệu Vector (FAISS)
│   ├── retrieval/          # Logic công cụ tìm kiếm (Search Engine)
│   └── visualization/      # Tạo XAI và bản đồ nhiệt
├── scripts/                # Các script tiện ích và demo
│   ├── build_index.py      # Script xây dựng database vector
│   └── demo_evis.py        # Script chạy demo toàn trình
├── data/                   # Thư mục chứa hình ảnh và file index
├── results/                # Kết quả tìm kiếm và hình ảnh XAI
├── plans/                  # Tài liệu thiết kế kiến trúc hệ thống
└── tests/                  # Các bài kiểm tra (Unit & Integration tests)
```

## 📖 Hướng dẫn sử dụng chi tiết

### Bước 1: Chuẩn bị dữ liệu
Dữ liệu đầu vào cần bao gồm tệp hình ảnh và metadata tương ứng. Metadata bao gồm:
- `event_name`: Tên sự kiện.
- `timestamp`: Thời gian (định dạng ISO8601).
- `lat`, `lon`: Tọa độ địa lý.

### Bước 2: Lập chỉ mục (Indexing)
Chạy script sau để trích xuất đặc trưng và xây dựng cơ sở dữ liệu vector:

```bash
python scripts/build_index.py
```
Script này sẽ quét các ảnh trong thư mục dữ liệu, thực hiện OCR, trích xuất đặc trưng CLIP, và lưu file index vào thư mục `data/`.

### Bước 3: Chạy Demo và Tìm kiếm
Sử dụng script demo để trải nghiệm khả năng tìm kiếm và xem kết quả giải thích:

```bash
python scripts/demo_evis.py
```
**Kết quả của Demo:**
- Thực hiện các truy vấn như: "coffee in Saigon", "Samsung store", "Tech Expo event".
- Kết quả tìm kiếm sẽ hiển thị trong terminal kèm theo điểm số tương đồng (Score).
- Các bản đồ nhiệt giải thích (XAI Heatmaps) sẽ được lưu tại `results/demo/`, giúp bạn hiểu tại sao ảnh đó lại khớp với truy vấn.

## 💻 Ví dụ về cách sử dụng Code

Dưới đây là cách sử dụng lớp `SearchEngine` trong code của bạn:

```python
from src.retrieval.search_engine import SearchEngine

# 1. Khởi tạo công cụ tìm kiếm với file index đã tạo
engine = SearchEngine(vector_db_path="data/evis_index")

# 2. Thực hiện truy vấn
results = engine.search("coffee in Saigon", top_k=5)

for res in results:
    print(f"Ảnh: {res['image_path']}, Score: {res['score']}")

# 3. Tạo giải thích (Heatmap) cho kết quả tốt nhất
engine.explain_result(results[0], "coffee in Saigon", output_dir="results")
```

## 🏗️ Kiến trúc hệ thống

EVIS tuân theo thiết kế mô-đun:
1. **Extraction**: Sử dụng CLIP Vision cho ảnh, CLIP Text cho tên sự kiện, và EasyOCR cho văn bản trong ảnh.
2. **Context Encoding**: Mã hóa thời gian/địa điểm bằng hàm SIN/COS kết hợp với đặc trưng tên sự kiện.
3. **Fusion Layer**: Cơ chế **Cross-Attention** nơi các đặc trưng hình ảnh truy vấn các đặc trưng văn bản và ngữ cảnh.
4. **Retrieval**: Tìm kiếm vector dựa trên FAISS.