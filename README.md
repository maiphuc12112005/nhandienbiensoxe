# 🚗 Hệ Thống Nhận Diện Biển Số Xe
**License Plate Recognition System — Full CV Pipeline**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org)

---

## 📋 Tổng Quan Pipeline

| Chương | Kỹ thuật | Triển khai |
|--------|----------|------------|
| **Ch.3** | Feature detection & matching | ORB keypoints, Hough Lines |
| **Ch.3** | Edge & line detection | Canny, Sobel, HoughLinesP |
| **Ch.4** | Image segmentation | K-Means (k=6), Mean-Shift, Watershed |
| **Ch.4** | Boundary detection | Canny + Sobel + Contour |
| **Ch.5** | Object recognition | EasyOCR + Tesseract + Haar Cascade |

---

## 📸 Dataset: RodoSol-ALPR (từ Kaggle)

### Giới thiệu

Dự án sử dụng **dataset RodoSol-ALPR** từ Kaggle — bộ dữ liệu biển số xe **thực tế**
được công bố trong bài báo khoa học:

> R. Laroca et al., *"On the Cross-dataset Generalization in License Plate Recognition"*, VISAPP 2022

**Nguồn**: [Kaggle ALPR Dataset](https://www.kaggle.com/datasets/mgmitesh/automatic-license-plate-recognition-alpr-dataset)

### Thông tin dataset

| Thuộc tính | Giá trị |
|-----------|---------|
| Tổng ảnh | **~24,238 ảnh** |
| Tập huấn luyện | **21,173 ảnh** |
| Tập kiểm tra | **1,019 ảnh** |
| Tập xác thực | **2,046 ảnh** |
| Nguồn | Camera tĩnh tại trạm thu phí cao tốc ES-060, Brazil |
| Độ phân giải | 1280 × 720 pixels |
| Loại phương tiện | Ô tô, xe máy, xe bus, xe tải |
| Điều kiện | Ban ngày & ban đêm, trời quang & mưa |
| Layout biển số | Brazilian (ABC-1234) & Mercosur (ABC1D23) |
| Định dạng label | `.txt` file (YOLO format) |
| Thư mục lưu trữ | `Data/train/`, `Data/test/`, `Data/valid/` |

### Cấu trúc dataset

```
Data/
├── train/
│   ├── images/     (21,173 ảnh)
│   └── labels/     (21,173 .txt files)
├── test/
│   ├── images/     (1,019 ảnh)
│   └── labels/     (1,019 .txt files)
└── valid/
    ├── images/     (2,046 ảnh)
    └── labels/     (2,046 .txt files)
```

### Cách lấy dataset
Tải từ Kaggle: https://www.kaggle.com/datasets/mgmitesh/automatic-license-plate-recognition-alpr-dataset
> Giải nén vào thư mục `Data/` của dự án

---

## ⚙️ Cài Đặt

```bash
# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Cài thư viện
pip install -r requirements.txt

# Linux: cài thêm Tkinter + Tesseract
sudo apt install python3-tk tesseract-ocr tesseract-ocr-vie
```

---

## 🚀 Chạy Chương Trình

### Giao diện GUI
```bash
python main.py
```
Mở giao diện GUI để phân tích ảnh biển số xe.

### Phân tích một ảnh bất kỳ
```bash
python main.py --image path/to/plate.jpg
```
Xuất chi tiết nhận diện: keypoints, cạnh, biển số, OCR kết quả.

### Thí nghiệm trên dataset *(yêu cầu dataset trong thư mục Data/)*
```bash
python main.py --kaggle              # Sử dụng ảnh từ dataset
python main.py --kaggle --n 100      # Thử nghiệm 100 ảnh
```
→ Xuất báo cáo JSON: `results/experiment_*.json`

### Chạy test suite
```bash
python utils/test_pipeline.py
```

---

## 🏗️ Kiến Trúc Pipeline

```
Input (Ảnh từ Data/ hoặc Camera)
        │
        ▼
┌──────────────────────────────────┐
│  Stage 1: CH.3 - Feature         │
│  • ORB Keypoint Detection (500)  │
│  • Canny Edge (threshold 50/150) │
│  • Sobel Gradient (ksize=3)      │
│  • Hough Lines (rho=1, thr=80)   │
└─────────────┬────────────────────┘
              ▼
┌──────────────────────────────────┐
│  Stage 2: CH.4 - Segmentation    │
│  • K-Means Clustering (k=6)      │
│  • Mean-Shift (sp=21, sr=51)     │
│  • Watershed Transform           │
│  • Contour + Biên Detection      │
└─────────────┬────────────────────┘
              ▼
┌──────────────────────────────────┐
│  Stage 3: Plate Localization     │
│  • Haar Cascade Classifier       │
│  • Geometric Contour Analysis    │
│  • Morphological Operations      │
│  • NMS Fusion (3 methods)        │
└─────────────┬────────────────────┘
              ▼
┌──────────────────────────────────┐
│  Stage 4: CH.5 - Recognition     │
│  • CLAHE Enhancement             │
│  • Otsu Thresholding             │
│  • EasyOCR (primary)             │
│  • Tesseract PSM 6/7/8/13        │
│  • Post-processing               │
└─────────────┬────────────────────┘
              ▼
    results/ → *_main.jpg  *_debug.jpg
               *_plate*.jpg  *.json
```

---

## 📁 Cấu Trúc Dự Án

```
license_plate_recognition/
├── main.py                      # Entry point
├── requirements.txt
├── README.md
├── src/
│   ├── pipeline.py              # Core CV pipeline
│   └── gui.py                   # GUI Tkinter
├── utils/
│   ├── create_real_dataset.py   # Dataset utilities
│   ├── create_samples.py        # Sample generation
│   ├── load_kaggle_alpr.py      # Dataset loader
│   └── test_pipeline.py         # Test suite (18 tests)
├── Data/
│   ├── train/                   # 21,173 training images + labels
│   ├── test/                    # 1,019 test images + labels
│   └── valid/                   # 2,046 validation images + labels
├── archive/
│   └── Data/                    # Backup dataset
└── results/
    ├── *_main.jpg
    ├── *_debug.jpg
    ├── *_plate*.jpg
    └── sample_report.json
```

---

## 🔍 Chi Tiết Components

### Pipeline.py
- **FeatureDetector**: ORB, SIFT, Canny, Sobel, HoughLines
- **SegmentationEngine**: K-Means, Mean-Shift, Watershed
- **PlateDetector**: Haar Cascade + Contour analysis
- **OCREngine**: EasyOCR + Tesseract + threshold-based

### GUI.py
- Upload ảnh từ file hoặc camera
- Hiển thị kết quả nhận diện theo từng stage
- Lưu kết quả chi tiết

### Main.py
- CLI interface với các tùy chọn experiment
- Xử lý từng batch ảnh từ dataset

---

## ❗ Xử Lý Sự Cố

| Vấn đề | Giải pháp |
|--------|----------|
| Dataset không tìm thấy | Kiểm tra thư mục `Data/` có tồn tại |
| `ModuleNotFoundError: easyocr` | `pip install easyocr` |
| `_tkinter.TclError` | `sudo apt install python3-tk` (Linux) |
| Tesseract not found | `sudo apt install tesseract-ocr` (Linux) |
| Camera không mở | Thử Camera ID: 0, 1, 2 |
| Không tìm được biển số | Điều chỉnh Canny threshold hoặc Haar Cascade parameters |

---

## 📊 Kết Quả

Kết quả sẽ được lưu tại `results/`:
- `*_main.jpg`: Ảnh gốc với biển số được highlight
- `*_debug.jpg`: Hình ảnh chi tiết các stage xử lý
- `*_plate*.jpg`: Patch biển số sau khi crop
- `*.json`: Report kết quả OCR chi tiết

---

## 📄 Thông Tin Dự Án

Dự án này xây dựng một hệ thống **nhận diện biển số xe hoàn chỉnh** sử dụng các kỹ thuật **Computer Vision** cổ điển kết hợp với **Deep Learning** (EasyOCR):

- **Xử lý ảnh**: Canny, Sobel, morphological operations
- **Phát hiện đặc trưng**: ORB keypoints, Hough Lines
- **Phân đoạn**: K-Means, Mean-Shift, Watershed
- **Định vị biển số**: Haar Cascade + Contour analysis
- **Nhận diện ký tự**: EasyOCR + Tesseract + threshold-based

### Tài Liệu Tham Khảo

- **OpenCV Documentation**: https://docs.opencv.org
- **EasyOCR**: https://github.com/JaidedAI/EasyOCR
- **Tesseract OCR**: https://github.com/UB-Mannheim/tesseract/wiki

---

## 👨‍💻 Hướng Dẫn Sử Dụng

1. **Cài đặt môi trường**: Theo phần [⚙️ Cài Đặt](#cài-đặt)
2. **Chuẩn bị dataset**: Dataset đã có sẵn trong thư mục `Data/`
3. **Chạy GUI**: `python main.py`
4. **Test nghiệm**: `python main.py --image path/to/test.jpg`

---

**Tạo bởi**: Tài Liệu Dự Án Nhận Diện Biển Số Xe  
**Ngôn ngữ**: Python 3.9+  
**License**: CC0 Public Domain
