"""
Tạo ảnh mẫu biển số xe Việt Nam để test pipeline
Dữ liệu thực tế: Biển số VN theo quy chuẩn 2023
"""

import cv2
import numpy as np
import os
import sys

def create_sample_plate(plate_text, filename, background_scenario='road'):
    """Tạo ảnh biển số giả lập thực tế"""
    # Kích thước biển số theo tiêu chuẩn VN (520x110mm, tỉ lệ ~4.7:1)
    pw, ph = 520, 110
    plate = np.ones((ph, pw, 3), dtype=np.uint8) * 255

    # Viền xanh biển số VN
    cv2.rectangle(plate, (0, 0), (pw-1, ph-1), (0, 100, 200), 4)
    cv2.rectangle(plate, (4, 4), (pw-5, ph-5), (0, 80, 180), 2)

    # Chữ đỏ dòng trên (tỉnh/thành)
    parts = plate_text.split('-')
    if len(parts) >= 2:
        top_text = parts[0]
        bottom_text = '-'.join(parts[1:])
    else:
        top_text = plate_text[:3]
        bottom_text = plate_text[3:]

    # Font scale
    font = cv2.FONT_HERSHEY_DUPLEX
    scale_top = 1.8
    scale_bot = 2.2
    thickness = 4

    # Dòng trên: mã tỉnh
    (tw, th), _ = cv2.getTextSize(top_text, font, scale_top, thickness)
    tx = (pw - tw) // 2
    cv2.putText(plate, top_text, (tx, 55), font, scale_top, (0, 0, 180), thickness)

    # Dòng dưới: số đăng ký
    (tw2, th2), _ = cv2.getTextSize(bottom_text, font, scale_bot, thickness)
    tx2 = (pw - tw2) // 2
    cv2.putText(plate, bottom_text, (tx2, 100), font, scale_bot, (0, 0, 180), thickness + 1)

    # Thêm nhiễu thực tế
    noise = np.random.randint(0, 15, plate.shape, dtype=np.uint8)
    plate = cv2.add(plate, noise)

    # Tạo ảnh toàn cảnh (giả lập chụp xe ngoài đường)
    if background_scenario == 'road':
        bg_h, bg_w = 400, 640
        bg = np.zeros((bg_h, bg_w, 3), dtype=np.uint8)
        # Đường
        bg[:] = (80, 80, 80)
        # Vạch kẻ đường
        for x in range(0, bg_w, 40):
            cv2.rectangle(bg, (x, bg_h//2 - 3), (x+25, bg_h//2 + 3), (255, 255, 255), -1)

        # Xe (hình chữ nhật đơn giản)
        car_x, car_y = 100, 150
        car_w, car_h = 440, 200
        cv2.rectangle(bg, (car_x, car_y), (car_x + car_w, car_y + car_h), (50, 80, 120), -1)
        cv2.rectangle(bg, (car_x + 30, car_y + 10), (car_x + car_w - 30, car_y + 90),
                      (150, 180, 200), -1)

        # Dán biển số lên xe
        plate_x = car_x + (car_w - pw // 2) // 2
        plate_y = car_y + car_h - ph // 2 - 20
        plate_small = cv2.resize(plate, (pw // 2, ph // 2))
        bg[plate_y:plate_y + ph//2, plate_x:plate_x + pw//2] = plate_small

    elif background_scenario == 'closeup':
        bg_h, bg_w = 200, 650
        bg = np.ones((bg_h, bg_w, 3), dtype=np.uint8) * 100
        plate_small = cv2.resize(plate, (pw, ph))
        py = (bg_h - ph) // 2
        px = (bg_w - pw) // 2
        if py >= 0 and px >= 0:
            bg[py:py+ph, px:px+pw] = plate_small

    else:
        bg = plate.copy()

    # Thêm motion blur nhẹ (giả lập xe đang chạy)
    kernel_size = 3
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size//2, :] = 1.0 / kernel_size
    bg = cv2.filter2D(bg, -1, kernel)

    return bg


def create_all_samples(output_dir="samples"):
    os.makedirs(output_dir, exist_ok=True)

    samples = [
        ("51A-123.45", "hanoi_sedan_road"),
        ("29B-678.90", "hcmc_suv_road"),
        ("51F-555.66", "hcmc_plate_closeup"),
        ("92H-234.56", "danang_truck_road"),
        ("30E-888.99", "hanoi_moto_closeup"),
        ("43A-112.23", "danang_car_road"),
        ("79C-333.44", "vungtau_boat_road"),
    ]

    print(f"Tạo {len(samples)} ảnh mẫu biển số VN...")
    created = []
    for plate_text, name in samples:
        scenario = 'closeup' if 'closeup' in name else 'road'
        img = create_sample_plate(plate_text, name, scenario)
        path = os.path.join(output_dir, f"{name}.jpg")
        cv2.imwrite(path, img)
        created.append(path)
        print(f"  ✓ {path}")

    print(f"\n✓ Đã tạo {len(created)} ảnh mẫu trong thư mục '{output_dir}'")
    return created


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "samples"
    create_all_samples(output)
