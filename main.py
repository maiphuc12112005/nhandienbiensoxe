#!/usr/bin/env python3
"""
License Plate Recognition System
==================================
Entry point chính

Usage:
    python main.py                          # Chạy GUI
    python main.py --image path/to/img.jpg  # Phân tích một ảnh
    python main.py --kaggle                 # Thí nghiệm trên Kaggle ALPR dataset  ← MỚI
    python main.py --kaggle --n 100         # Thí nghiệm với 100 ảnh               ← MỚI
    python main.py --build-dataset          # Tạo dataset tổng hợp dự phòng
    python main.py --demo                   # Demo nhanh
"""

import argparse, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

RESULTS_DIR = "results"
DATASET_DIR = "dataset/real_images"


def run_gui():
    try:
        import tkinter as tk
    except ImportError:
        print("✗ Cài Tkinter: sudo apt install python3-tk")
        sys.exit(1)
    from src.gui import LicensePlateGUI
    root = tk.Tk()
    app  = LicensePlateGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    print("✓ Khởi động GUI...")
    root.mainloop()


def run_cli(image_path, output_dir=RESULTS_DIR):
    from src.pipeline import LicensePlateSystem
    import cv2
    print("→ Khởi tạo hệ thống...")
    system = LicensePlateSystem(output_dir=output_dir)
    print(f"→ Phân tích: {image_path}")
    result = system.process_image(image_path, save_result=True)
    plates = result['plates']
    print(f"\n{'='*55}")
    print(f"  Keypoints (ORB): {len(result['keypoints'])}")
    print(f"  Hough lines    : {result['lines_count']}")
    print(f"  Biển số tìm thấy: {len(plates)}")
    for i, p in enumerate(plates):
        print(f"  #{i+1}: {p['text']}  ({p['confidence']:.1%})")
    if result['saved_files']:
        print(f"  Kết quả: {result['saved_files'].get('main','')}")
        print(f"  Debug  : {result['saved_files'].get('debug','')}")
    print(f"{'='*55}\n")


def run_kaggle_experiment(max_images=50, output_dir=RESULTS_DIR):
    """Chạy thí nghiệm trên dataset Kaggle ALPR (thay thế RodoSol)"""
    from utils.load_kaggle_alpr import run_experiment_kaggle
    run_experiment_kaggle(max_images=max_images, output_dir=output_dir)


def build_synthetic_dataset(n=30):
    """Tạo dataset tổng hợp dự phòng (khi chưa tải Kaggle dataset)"""
    from utils.create_real_dataset import create_full_dataset
    print(f"→ Tạo {n} ảnh dataset tổng hợp...")
    create_full_dataset(DATASET_DIR, n)


def run_demo():
    from src.pipeline import LicensePlateSystem
    import cv2

    # Ưu tiên dùng Kaggle ALPR nếu có
    from utils.load_kaggle_alpr import check_dataset_available, load_kaggle_alpr_subset
    if check_dataset_available():
        print("→ Demo với dataset Kaggle ALPR thực tế...\n")
        samples = load_kaggle_alpr_subset(None, max_images=3)
        system  = LicensePlateSystem(output_dir=RESULTS_DIR)
        for s in samples:
            res    = system.process_image(s['image'], save_result=True, filename_prefix='demo')
            plates = res['plates']
            best   = max(plates, key=lambda p: p['confidence']) if plates else None
            gt     = s['plate_text'] if s['plate_text'] not in ('', 'UNKNOWN') else '(unknown)'
            print(f"  GT   : {gt}")
            print(f"  Pred : {best['text'] if best else 'Không tìm thấy'}")
            print(f"  Conf : {best['confidence']:.1%}\n" if best else "  Conf : --\n")
    else:
        # Fallback: dataset tổng hợp
        if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
            build_synthetic_dataset(5)
        imgs   = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith('.jpg')])[:3]
        system = LicensePlateSystem(output_dir=RESULTS_DIR)
        print("→ Demo với dataset tổng hợp (Kaggle ALPR chưa có)...\n")
        for fname in imgs:
            img = cv2.imread(os.path.join(DATASET_DIR, fname))
            res = system.process_image(img, save_result=True, filename_prefix='demo')
            plates = res['plates']
            best   = max(plates, key=lambda p: p['confidence']) if plates else None
            print(f"  File : {fname}")
            print(f"  Pred : {best['text'] if best else 'Không tìm thấy'}")
            print(f"  Conf : {best['confidence']:.1%}\n" if best else "  Conf : --\n")

    print("✓ Xem ảnh kết quả trong results/")


def main():
    parser = argparse.ArgumentParser(
        description="License Plate Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--image',         '-i', type=str, help='Đường dẫn ảnh (CLI mode)')
    parser.add_argument('--output',        '-o', type=str, default=RESULTS_DIR)

    # ── Dataset Kaggle ALPR (MỚI) ──────────────────────────────────────────────
    parser.add_argument('--kaggle',
                        action='store_true',
                        help='Thí nghiệm trên dataset Kaggle ALPR')
    parser.add_argument('--kaggle-dir',
                        type=str, default=None,
                        help='Đường dẫn tới thư mục dataset Kaggle (mặc định: dataset/kaggle-alpr/)')
    parser.add_argument('--n',             type=int, default=50,
                        help='Số ảnh thí nghiệm (mặc định: 50)')

    parser.add_argument('--build-dataset', action='store_true',
                        help='Tạo dataset tổng hợp dự phòng')
    parser.add_argument('--dataset-size',  type=int, default=30)
    parser.add_argument('--demo',          action='store_true')

    args = parser.parse_args()

    if args.kaggle:
        # Nếu chỉ định thư mục, truyền thẳng vào
        if args.kaggle_dir:
            from utils.load_kaggle_alpr import run_experiment_kaggle
            run_experiment_kaggle(
                max_images=args.n,
                output_dir=args.output,
                dataset_dir=args.kaggle_dir
            )
        else:
            run_kaggle_experiment(args.n, args.output)

    elif args.build_dataset:
        build_synthetic_dataset(args.dataset_size)

    elif args.demo:
        run_demo()

    elif args.image:
        if not os.path.exists(args.image):
            print(f"✗ Không tìm thấy: {args.image}")
            sys.exit(1)
        run_cli(args.image, args.output)

    else:
        run_gui()


if __name__ == "__main__":
    main()