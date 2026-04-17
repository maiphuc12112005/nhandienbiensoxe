"""
load_kaggle_alpr.py
===================
Tích hợp dataset từ Kaggle:
  https://www.kaggle.com/datasets/mgmitesh/automatic-license-plate-recognition-alpr-dataset

CÁCH TẢI DATASET:
  Cách 1 - Kaggle Web:
    1. Vào link trên → Download (nút góc phải)
    2. Giải nén → đặt vào thư mục: dataset/kaggle-alpr/

  Cách 2 - Kaggle CLI (nhanh hơn):
    pip install kaggle
    kaggle datasets download -d mgmitesh/automatic-license-plate-recognition-alpr-dataset
    unzip automatic-license-plate-recognition-alpr-dataset.zip -d dataset/kaggle-alpr/

CẤU TRÚC THƯ MỤC HỖ TRỢ (tự động nhận diện):

  Kiểu A - ảnh + XML (Pascal VOC):
    dataset/kaggle-alpr/
    ├── images/         (hoặc Cars/, train/, ...)
    │   ├── img001.jpg
    │   └── ...
    └── annotations/    (hoặc labels/, xmls/, ...)
        ├── img001.xml
        └── ...

  Kiểu B - ảnh + CSV:
    dataset/kaggle-alpr/
    ├── images/
    │   └── *.jpg
    └── labels.csv      (cột: filename, plate / text / label)

  Kiểu C - ảnh đặt thẳng trong thư mục (không có annotation):
    dataset/kaggle-alpr/
    └── *.jpg  (tên file chứa biển số, ví dụ: ABC1234.jpg)
"""

import os
import cv2
import csv
import json
import glob
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ── Đường dẫn dataset ──────────────────────────────────────────────────────────
KAGGLE_ALPR_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'kaggle-alpr'
)

# Tên các thư mục ảnh và annotation thường gặp trong dataset ALPR Kaggle
_IMG_DIR_NAMES  = ['images', 'image', 'imgs', 'Cars', 'cars', 'train', 'test', 'val', '.']
_ANN_DIR_NAMES  = ['annotations', 'annotation', 'labels', 'label', 'xmls', 'xml', 'Annotations']
_CSV_NAMES      = ['labels.csv', 'annotations.csv', 'plates.csv', 'data.csv',
                   'train_labels.csv', 'test_labels.csv', '_annotations.csv']


# ── Kiểm tra dataset ───────────────────────────────────────────────────────────
def check_dataset_available(dataset_dir=None):
    """Kiểm tra dataset Kaggle ALPR đã tải về chưa."""
    d = dataset_dir or KAGGLE_ALPR_DIR
    if not os.path.isdir(d):
        return False
    # Có ít nhất 1 file ảnh bất kỳ trong cây thư mục
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        if glob.glob(os.path.join(d, '**', ext), recursive=True):
            return True
    return False


# ── Tìm thư mục ảnh / annotation ──────────────────────────────────────────────
def _find_img_dir(base):
    for name in _IMG_DIR_NAMES:
        p = os.path.join(base, name)
        if os.path.isdir(p):
            imgs = glob.glob(os.path.join(p, '*.jpg')) + \
                   glob.glob(os.path.join(p, '*.jpeg')) + \
                   glob.glob(os.path.join(p, '*.png'))
            if imgs:
                return p
    # Thư mục gốc
    imgs = glob.glob(os.path.join(base, '*.jpg')) + \
           glob.glob(os.path.join(base, '*.jpeg')) + \
           glob.glob(os.path.join(base, '*.png'))
    if imgs:
        return base
    # Tìm đệ quy
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                return root
    return None


def _find_ann_dir(base):
    for name in _ANN_DIR_NAMES:
        p = os.path.join(base, name)
        if os.path.isdir(p):
            xmls = glob.glob(os.path.join(p, '*.xml'))
            if xmls:
                return p, 'xml'
    return None, None


def _find_csv(base):
    for name in _CSV_NAMES:
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    # Tìm bất kỳ file csv
    for f in glob.glob(os.path.join(base, '**', '*.csv'), recursive=True):
        return f
    return None


# ── Parse XML (Pascal VOC) ─────────────────────────────────────────────────────
def parse_xml_annotation(xml_path):
    """
    Đọc file XML annotation kiểu Pascal VOC.
    Trả về dict: {plate_text, xmin, ymin, xmax, ymax}
    """
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        root = tree.getroot()

        result = {'plate_text': '', 'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 0}

        # Lấy bounding box
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                result['xmin'] = int(float(bndbox.findtext('xmin', 0)))
                result['ymin'] = int(float(bndbox.findtext('ymin', 0)))
                result['xmax'] = int(float(bndbox.findtext('xmax', 0)))
                result['ymax'] = int(float(bndbox.findtext('ymax', 0)))

            # Lấy text biển số nếu có
            for tag in ('name', 'plate', 'text', 'label'):
                val = obj.findtext(tag, '')
                if val and val.lower() not in ('license plate', 'plate', 'car', 'vehicle', ''):
                    result['plate_text'] = val.strip()
                    break

        # Nếu text nằm ở root
        if not result['plate_text']:
            for tag in ('plate', 'text', 'label', 'number'):
                val = root.findtext(tag, '')
                if val:
                    result['plate_text'] = val.strip()
                    break

        return result
    except Exception as e:
        return {'plate_text': '', 'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 0}


# ── Parse CSV ──────────────────────────────────────────────────────────────────
def _load_csv_labels(csv_path):
    """
    Đọc CSV label. Trả về dict: {filename -> {plate_text, xmin, ymin, xmax, ymax}}
    Tự động nhận dạng tên cột.
    """
    labels = {}
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            cols = [c.lower().strip() for c in (reader.fieldnames or [])]

            # Map tên cột thực tế
            col_file = next((reader.fieldnames[i] for i, c in enumerate(cols)
                             if c in ('filename', 'file', 'image', 'image_name', 'name')), None)
            col_text = next((reader.fieldnames[i] for i, c in enumerate(cols)
                             if c in ('plate', 'text', 'label', 'plate_text',
                                      'number', 'license_plate', 'value')), None)
            col_xmin = next((reader.fieldnames[i] for i, c in enumerate(cols)
                             if c in ('xmin', 'x_min', 'left')), None)
            col_ymin = next((reader.fieldnames[i] for i, c in enumerate(cols)
                             if c in ('ymin', 'y_min', 'top')), None)
            col_xmax = next((reader.fieldnames[i] for i, c in enumerate(cols)
                             if c in ('xmax', 'x_max', 'right')), None)
            col_ymax = next((reader.fieldnames[i] for i, c in enumerate(cols)
                             if c in ('ymax', 'y_max', 'bottom')), None)

            for row in reader:
                fname = os.path.basename(row[col_file].strip()) if col_file else ''
                labels[fname] = {
                    'plate_text': row[col_text].strip() if col_text else '',
                    'xmin': int(float(row[col_xmin])) if col_xmin else 0,
                    'ymin': int(float(row[col_ymin])) if col_ymin else 0,
                    'xmax': int(float(row[col_xmax])) if col_xmax else 0,
                    'ymax': int(float(row[col_ymax])) if col_ymax else 0,
                }
    except Exception as e:
        print(f"  [CSV parse error] {e}")
    return labels


# ── Hàm load chính ─────────────────────────────────────────────────────────────
def load_kaggle_alpr_subset(subset=None, max_images=100, dataset_dir=None):
    """
    Load ảnh từ dataset Kaggle ALPR.

    Args:
        subset    : 'train' | 'test' | 'val' | None (lấy tất cả)
        max_images: số ảnh tối đa
        dataset_dir: đường dẫn đến thư mục dataset (mặc định: dataset/kaggle-alpr/)

    Returns:
        list of dict: [{image, path, filename, plate_text, bbox, vehicle}, ...]
    """
    base = dataset_dir or KAGGLE_ALPR_DIR

    if not check_dataset_available(base):
        _print_download_guide()
        return []

    print(f"→ Đang load dataset từ: {base}")

    # Nếu có thư mục subset (train/test/val)
    if subset:
        for name in _IMG_DIR_NAMES:
            sub_path = os.path.join(base, name, subset)
            if os.path.isdir(sub_path):
                base_search = sub_path
                break
            sub_path2 = os.path.join(base, subset)
            if os.path.isdir(sub_path2):
                base_search = sub_path2
                break
        else:
            base_search = base
    else:
        base_search = base

    # Tìm thư mục ảnh
    img_dir = _find_img_dir(base_search)
    if img_dir is None:
        print(f"✗ Không tìm thấy ảnh trong: {base_search}")
        return []

    # Thu thập danh sách ảnh
    img_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
        img_files.extend(glob.glob(os.path.join(img_dir, ext)))
    img_files = sorted(img_files)[:max_images]

    if not img_files:
        print(f"✗ Không có ảnh trong: {img_dir}")
        return []

    # Xác định nguồn label (XML / CSV / từ tên file)
    ann_dir, ann_type = _find_ann_dir(base_search)
    if ann_dir is None:
        ann_dir, ann_type = _find_ann_dir(base)

    csv_labels = {}
    csv_path = _find_csv(base_search) or _find_csv(base)
    if csv_path:
        csv_labels = _load_csv_labels(csv_path)
        print(f"  Đọc CSV labels: {csv_path}  ({len(csv_labels)} mục)")
    elif ann_dir:
        print(f"  Đọc XML annotations: {ann_dir}/")
    else:
        print("  Không có annotations — đọc tên file làm label")

    # Load từng ảnh
    data = []
    for img_path in img_files:
        img = cv2.imread(img_path)
        if img is None:
            continue

        fname     = os.path.basename(img_path)
        stem      = os.path.splitext(fname)[0]
        plate_txt = ''
        bbox      = (0, 0, img.shape[1], img.shape[0])  # toàn ảnh nếu không có bbox

        # Lấy label từ CSV
        if fname in csv_labels:
            info = csv_labels[fname]
            plate_txt = info['plate_text']
            if info['xmax'] > info['xmin'] and info['ymax'] > info['ymin']:
                bbox = (info['xmin'], info['ymin'],
                        info['xmax'] - info['xmin'],
                        info['ymax'] - info['ymin'])

        # Lấy label từ XML
        elif ann_dir:
            xml_path = os.path.join(ann_dir, stem + '.xml')
            if os.path.exists(xml_path):
                info = parse_xml_annotation(xml_path)
                plate_txt = info['plate_text']
                if info['xmax'] > info['xmin']:
                    bbox = (info['xmin'], info['ymin'],
                            info['xmax'] - info['xmin'],
                            info['ymax'] - info['ymin'])

        # Đọc từ tên file (ví dụ: ABC1234_001.jpg → ABC1234)
        if not plate_txt:
            plate_txt = stem.split('_')[0].split('-')[0].upper()

        data.append({
            'image':      img,
            'path':       img_path,
            'filename':   fname,
            'plate_text': plate_txt,
            'bbox':       bbox,
            'vehicle':    'car',
            'layout':     'unknown',
        })

    print(f"✓ Load xong {len(data)} ảnh từ Kaggle ALPR dataset")
    return data


# ── Crop biển số theo bbox ─────────────────────────────────────────────────────
def crop_plate_from_bbox(image, bbox):
    """
    Cắt vùng biển số từ bounding box (x, y, w, h).
    """
    x, y, w, h = [int(v) for v in bbox]
    H, W = image.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    if x2 > x1 and y2 > y1:
        return image[y1:y2, x1:x2]
    return None


# ── Chạy thí nghiệm chính ──────────────────────────────────────────────────────
def run_experiment_kaggle(max_images=50, output_dir='results', dataset_dir=None):
    """
    Chạy pipeline CV trên dataset Kaggle ALPR và xuất báo cáo.
    Đây là hàm CHÍNH để thực hiện thí nghiệm.
    """
    import time
    from datetime import datetime

    base = dataset_dir or KAGGLE_ALPR_DIR

    if not check_dataset_available(base):
        _print_download_guide()
        return None

    # Import pipeline từ src/
    try:
        from pipeline import LicensePlateSystem
    except ImportError:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from pipeline import LicensePlateSystem

    print(f"\n{'='*60}")
    print(f"  THÍ NGHIỆM TRÊN DATASET KAGGLE ALPR")
    print(f"  {base}")
    print(f"{'='*60}")

    system = LicensePlateSystem(output_dir=output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Thử load theo subset, nếu không có thì load tất cả
    samples = load_kaggle_alpr_subset('test',  max_images=max_images//2, dataset_dir=base)
    samples += load_kaggle_alpr_subset('train', max_images=max_images//2, dataset_dir=base)

    # Nếu không tìm thấy subset, load thẳng tất cả
    if not samples:
        samples = load_kaggle_alpr_subset(None, max_images=max_images, dataset_dir=base)

    if not samples:
        print("✗ Không load được ảnh nào!")
        return None

    print(f"\n→ Chạy pipeline trên {len(samples)} ảnh...\n")

    results  = []
    correct  = 0
    detected = 0
    t_start  = time.time()

    for i, sample in enumerate(samples):
        img      = sample['image']
        gt_plate = sample['plate_text'].upper().replace('-', '').replace(' ', '')

        t0  = time.time()
        res = system.process_image(
            img, save_result=True,
            filename_prefix=f"kaggle_{i+1:03d}"
        )
        dt = time.time() - t0

        plates = res['plates']
        if plates:
            detected += 1
            best = max(plates, key=lambda p: p['confidence'])
            pred = best['text'].upper().replace(' ', '').replace('-', '')
            conf = best['confidence']
        else:
            pred, conf = '', 0.0

        # So sánh với ground truth (nếu có)
        if gt_plate and gt_plate not in ('UNKNOWN', ''):
            is_correct = (pred == gt_plate) or \
                         (gt_plate in pred) or \
                         (pred in gt_plate and len(pred) >= 4)
            if is_correct:
                correct += 1
            gt_known = True
        else:
            is_correct = False
            gt_known   = False

        icon = '✓' if is_correct else ('~' if plates else '✗')
        gt_display = gt_plate if gt_known else '(unknown)'
        print(f"  [{i+1:03d}] {icon}  GT={gt_display:12s}  Pred={pred:12s}  conf={conf:.0%}  {dt:.1f}s")

        results.append({
            'id':           i + 1,
            'file':         sample['filename'],
            'vehicle':      sample['vehicle'],
            'ground_truth': gt_plate,
            'predicted':    pred,
            'confidence':   round(conf, 3),
            'detected':     bool(plates),
            'correct':      is_correct,
            'time_s':       round(dt, 2),
            'keypoints':    len(res['keypoints']),
            'lines':        res['lines_count'],
        })

    elapsed   = time.time() - t_start
    n         = len(samples)
    n_gt      = sum(1 for r in results if r['ground_truth'] not in ('', 'UNKNOWN'))
    det_rate  = detected / n * 100
    acc_rate  = (correct / n_gt * 100) if n_gt > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  KẾT QUẢ THÍ NGHIỆM")
    print(f"{'='*60}")
    print(f"  Tổng ảnh           : {n}")
    print(f"  Có ground truth    : {n_gt}")
    print(f"  Phát hiện biển số  : {detected}/{n}  ({det_rate:.1f}%)")
    if n_gt > 0:
        print(f"  Nhận dạng đúng     : {correct}/{n_gt}  ({acc_rate:.1f}%)")
    print(f"  Thời gian tổng     : {elapsed:.1f}s")
    print(f"  Thời gian/ảnh      : {elapsed/n:.2f}s")
    print(f"  OCR engine         : {system.recognizer.ocr_engine}")

    # Xuất báo cáo JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp":          datetime.now().isoformat(),
        "dataset":            "Kaggle ALPR Dataset",
        "dataset_source":     "https://www.kaggle.com/datasets/mgmitesh/automatic-license-plate-recognition-alpr-dataset",
        "total_images":       n,
        "images_with_gt":     n_gt,
        "detected":           detected,
        "detection_rate":     round(det_rate / 100, 3),
        "correct":            correct,
        "accuracy":           round(acc_rate / 100, 3) if n_gt > 0 else None,
        "total_time_s":       round(elapsed, 2),
        "avg_time_per_img_s": round(elapsed / n, 3),
        "ocr_engine":         system.recognizer.ocr_engine,
        "pipeline_stages":    [
            "ORB+Canny+Sobel+Hough",
            "KMeans+MeanShift+Watershed",
            "Cascade+Contour+Morpho",
            "Tesseract/EasyOCR"
        ],
        "results": results,
    }
    report_path = os.path.join(output_dir, f"kaggle_experiment_{ts}.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  Báo cáo JSON : {report_path}")
    print(f"  Ảnh kết quả  : {output_dir}/")
    print(f"{'='*60}\n")
    return report


# ── Hướng dẫn tải dataset ─────────────────────────────────────────────────────
def _print_download_guide():
    print("\n" + "="*60)
    print("  DATASET KAGGLE ALPR CHƯA CÓ")
    print("="*60)
    print("""
  Tải dataset theo 1 trong 2 cách:

  ── Cách 1: Tải thủ công ──────────────────────────────────
  1. Vào: https://www.kaggle.com/datasets/mgmitesh/
             automatic-license-plate-recognition-alpr-dataset
  2. Nhấn nút "Download" (cần tài khoản Kaggle)
  3. Giải nén file .zip vào thư mục:
       dataset/kaggle-alpr/

  ── Cách 2: Kaggle CLI (nhanh hơn) ───────────────────────
  pip install kaggle

  # Đặt file kaggle.json vào ~/.kaggle/kaggle.json
  # (Lấy tại: kaggle.com → Account → Create API Token)

  kaggle datasets download -d mgmitesh/automatic-license-plate-recognition-alpr-dataset
  mkdir -p dataset/kaggle-alpr
  unzip automatic-license-plate-recognition-alpr-dataset.zip -d dataset/kaggle-alpr/

  ── Sau khi tải xong ────────────────────────────────────
  python main.py --kaggle --n 50
""")
    print("="*60 + "\n")


# ── Chạy trực tiếp ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    max_imgs = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    out      = sys.argv[2] if len(sys.argv) > 2 else 'results'
    run_experiment_kaggle(max_images=max_imgs, output_dir=out)