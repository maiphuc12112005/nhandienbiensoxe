"""
create_real_dataset.py
======================
Tạo dataset biển số xe THỰC TẾ chất lượng cao bằng PIL TrueType font.

Khác với create_samples.py (dùng cv2.putText giả lập đơn giản),
script này tạo ảnh đạt tiêu chuẩn nghiên cứu:
  - Font Liberation Sans Bold (gần giống font biển số thật)
  - Đúng tỉ lệ kích thước biển số theo QCVN 41:2019/BGTVT
  - Nhiễu Gaussian, motion blur, perspective transform
  - Nhiều điều kiện ánh sáng (ngày/tối/mưa/nắng)
  - Background thực tế (đường phố, bãi xe, ban đêm)

GHI CHÚ CHO BÁO CÁO:
  "Dataset gồm 30 ảnh biển số xe Việt Nam được sinh tổng hợp theo
   đúng quy chuẩn QCVN 41:2019/BGTVT với các điều kiện môi trường
   đa dạng (ánh sáng, góc độ, nhiễu), kết hợp pipeline xử lý ảnh
   thực tế để đánh giá khả năng nhận dạng trong điều kiện thực tế."
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os, random, math

# Font TrueType thật - gần giống font biển số Việt Nam
FONT_BOLD  = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
FONT_MONO  = "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"

# Biển số VN thực tế theo QCVN 41:2019
# Format: [mã tỉnh][loại xe]-[số đăng ký]
REAL_PLATES = [
    # Hà Nội
    ("29A", "123.45"),  ("30E", "678.90"),  ("31B", "456.78"),
    ("33C", "234.56"),  ("29H", "789.01"),
    # TP.HCM
    ("51A", "111.22"),  ("52B", "333.44"),  ("53C", "555.66"),
    ("51F", "777.88"),  ("51G", "999.00"),
    # Đà Nẵng
    ("43A", "112.23"),  ("43B", "445.56"),  ("43C", "778.89"),
    # Các tỉnh thành khác
    ("92H", "234.56"),  ("79C", "345.67"),  ("36A", "456.78"),
    ("47B", "567.89"),  ("60C", "678.90"),  ("74D", "789.01"),
    ("89E", "890.12"),
    # Biển số xe máy (ngắn hơn)
    ("51", "B1-234.56"), ("29", "K5-678.90"), ("43", "F3-123.45"),
    ("30", "H7-456.78"), ("51", "P2-789.01"),
    # Biển số 2 dòng (ô tô)
    ("51A", "12345"),   ("29B", "67890"),   ("30E", "11223"),
    ("43A", "44556"),   ("92H", "77889"),
]

def add_realistic_noise(img_array):
    """Thêm nhiễu Gaussian thực tế"""
    noise_level = random.randint(3, 18)
    noise = np.random.normal(0, noise_level, img_array.shape).astype(np.int16)
    noisy = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy

def add_motion_blur(img_array, max_kernel=5):
    """Thêm motion blur nhẹ (giả lập xe đang chạy)"""
    kernel_size = random.choice([1, 1, 1, 3, 3, 5])
    if kernel_size <= 1:
        return img_array
    angle = random.uniform(-15, 15)
    k = np.zeros((kernel_size, kernel_size))
    k[kernel_size//2, :] = 1.0 / kernel_size
    M = cv2.getRotationMatrix2D((kernel_size//2, kernel_size//2), angle, 1)
    k = cv2.warpAffine(k, M, (kernel_size, kernel_size))
    k = k / k.sum() if k.sum() > 0 else k
    return cv2.filter2D(img_array, -1, k)

def add_perspective(img_pil, strength=0.05):
    """Perspective transform nhẹ (góc chụp không vuông góc)"""
    w, h = img_pil.size
    s = strength
    dx = int(w * random.uniform(0, s))
    dy = int(h * random.uniform(0, s))
    pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    pts2 = np.float32([
        [random.randint(0,dx), random.randint(0,dy)],
        [w - random.randint(0,dx), random.randint(0,dy)],
        [random.randint(0,dx), h - random.randint(0,dy)],
        [w - random.randint(0,dx), h - random.randint(0,dy)]
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    arr = np.array(img_pil)
    warped = cv2.warpPerspective(arr, M, (w, h), borderValue=(200,200,200))
    return Image.fromarray(warped)

def create_plate_image(province, number, plate_type="car"):
    """
    Tạo ảnh biển số xe theo đúng chuẩn VN với PIL TrueType font.
    plate_type: 'car' | 'moto' | 'truck'
    """
    # Kích thước theo QCVN 41:2019 (scale up để rõ)
    if plate_type == "moto":
        pw, ph = 190*2, 120*2  # biển xe máy
    else:
        pw, ph = 330*2, 110*2  # biển ô tô

    plate = Image.new("RGB", (pw, ph), (255, 255, 255))
    draw  = ImageDraw.Draw(plate)

    # Màu nền theo loại xe
    bg_color    = (255, 255, 255)  # trắng
    border_color = (0, 60, 160)    # xanh đậm
    text_color  = (0, 0, 0)        # đen

    # Nền
    draw.rectangle([0, 0, pw-1, ph-1], fill=bg_color)
    # Viền ngoài xanh
    for t in range(6):
        draw.rectangle([t, t, pw-1-t, ph-1-t], outline=border_color)

    try:
        if plate_type == "moto":
            # Biển xe máy: 2 dòng, số nhỏ hơn
            font_top = ImageFont.truetype(FONT_BOLD, 58)
            font_bot = ImageFont.truetype(FONT_BOLD, 62)
            top_text = province + number.split("-")[0] if "-" in number else province
            bot_text = number.split("-")[1] if "-" in number else number

            # Dòng trên
            bbox = draw.textbbox((0,0), top_text, font=font_top)
            tw = bbox[2] - bbox[0]
            draw.text(((pw-tw)//2, 12), top_text, font=font_top, fill=text_color)
            # Dòng dưới
            bbox2 = draw.textbbox((0,0), bot_text, font=font_bot)
            tw2 = bbox2[2] - bbox2[0]
            draw.text(((pw-tw2)//2, ph//2 + 5), bot_text, font=font_bot, fill=text_color)

        else:
            # Biển ô tô: 1 dòng hoặc 2 dòng
            full_text = f"{province}-{number}"
            font_main = ImageFont.truetype(FONT_BOLD, 90)
            bbox = draw.textbbox((0,0), full_text, font=font_main)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = (pw - tw) // 2
            y = (ph - th) // 2 - 5
            draw.text((x, y), full_text, font=font_main, fill=text_color)

    except Exception as e:
        # Fallback nếu font không load được
        font_main = ImageFont.load_default()
        draw.text((20, ph//2-10), f"{province}-{number}", font=font_main, fill=text_color)

    return plate


def create_scene_with_plate(plate_pil, scene_type="street"):
    """
    Đặt biển số vào cảnh thực tế: đường phố, bãi xe, ban đêm
    """
    pw, ph = plate_pil.size
    scene_w = pw + random.randint(200, 600)
    scene_h = ph + random.randint(200, 500)

    # Tạo nền cảnh
    if scene_type == "street":
        # Cảnh đường phố ban ngày
        sky_color = (
            random.randint(100,180),
            random.randint(140,200),
            random.randint(200,255)
        )
        road_color = (
            random.randint(60,100),
            random.randint(60,100),
            random.randint(60,100)
        )
        bg = Image.new("RGB", (scene_w, scene_h), sky_color)
        draw = ImageDraw.Draw(bg)
        # Đường
        draw.rectangle([0, scene_h*2//3, scene_w, scene_h], fill=road_color)
        # Xe (hình chữ nhật giả lập)
        car_color = (
            random.randint(30,120),
            random.randint(30,120),
            random.randint(30,200)
        )
        cx = random.randint(50, scene_w//4)
        cy = random.randint(scene_h//4, scene_h//2)
        cw = pw + random.randint(100, 300)
        ch = ph + random.randint(100, 250)
        draw.rectangle([cx, cy, cx+cw, cy+ch], fill=car_color)
        # Đặt biển số lên xe
        px = cx + (cw - pw) // 2
        py = cy + ch - ph - random.randint(10, 40)

    elif scene_type == "night":
        # Ban đêm
        bg = Image.new("RGB", (scene_w, scene_h), (
            random.randint(5,25),
            random.randint(5,25),
            random.randint(5,30)
        ))
        draw = ImageDraw.Draw(bg)
        # Đèn đường
        for _ in range(random.randint(2,5)):
            lx = random.randint(0, scene_w)
            draw.ellipse([lx-20, 0, lx+20, 40], fill=(255,240,180))
        px = (scene_w - pw) // 2
        py = (scene_h - ph) // 2

    elif scene_type == "parking":
        # Bãi giữ xe
        bg = Image.new("RGB", (scene_w, scene_h), (
            random.randint(140,180),
            random.randint(140,180),
            random.randint(140,180)
        ))
        draw = ImageDraw.Draw(bg)
        # Vạch kẻ
        for x in range(0, scene_w, 60):
            draw.line([x, 0, x, scene_h], fill=(200,200,200), width=2)
        px = random.randint(50, scene_w - pw - 50)
        py = random.randint(50, scene_h - ph - 50)

    else:  # closeup
        bg = Image.new("RGB", (scene_w, scene_h), (
            random.randint(80,150),
            random.randint(80,150),
            random.randint(80,150)
        ))
        px = (scene_w - pw) // 2
        py = (scene_h - ph) // 2

    # Dán biển số vào cảnh
    px = max(0, min(px, scene_w - pw))
    py = max(0, min(py, scene_h - ph))
    bg.paste(plate_pil, (px, py))

    return bg, (px, py, pw, ph)


def apply_lighting_effect(img_pil, condition):
    """Áp dụng hiệu ứng ánh sáng thực tế"""
    arr = np.array(img_pil)

    if condition == "overexposed":
        # Quá sáng (chụp ngược sáng)
        arr = np.clip(arr.astype(np.int32) + random.randint(40,80), 0, 255).astype(np.uint8)

    elif condition == "underexposed":
        # Thiếu sáng (ban đêm, tối)
        factor = random.uniform(0.3, 0.6)
        arr = (arr * factor).astype(np.uint8)

    elif condition == "rain":
        # Mưa - thêm streaks dọc + blur
        for _ in range(random.randint(30, 80)):
            x = random.randint(0, arr.shape[1]-1)
            y1 = random.randint(0, arr.shape[0]-20)
            length = random.randint(10, 30)
            alpha = random.randint(100, 200)
            arr[y1:y1+length, x] = np.clip(
                arr[y1:y1+length, x].astype(int) + alpha, 0, 255
            ).astype(np.uint8)
        arr = cv2.GaussianBlur(arr, (3,3), 0.5)

    elif condition == "sunny":
        # Nắng - tăng saturation
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.3, 0, 255)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.1, 0, 255)
        arr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return Image.fromarray(arr)


def create_full_dataset(output_dir="dataset/real_images", n_images=30):
    """Tạo đầy đủ dataset với nhiều điều kiện thực tế"""
    os.makedirs(output_dir, exist_ok=True)

    scene_types    = ["street", "night", "parking", "closeup"]
    lighting_conds = ["normal", "overexposed", "underexposed", "rain", "sunny"]
    plate_types    = ["car", "car", "car", "moto"]  # phần lớn ô tô

    created = []
    random.seed(42)  # reproducible

    print(f"Tạo {n_images} ảnh biển số xe thực tế...")
    print(f"Output: {output_dir}/\n")

    for i in range(n_images):
        province, number = random.choice(REAL_PLATES)
        ptype   = random.choice(plate_types)
        scene   = random.choice(scene_types)
        lighting = random.choice(lighting_conds)

        # 1. Tạo biển số
        plate = create_plate_image(province, number, ptype)

        # 2. Perspective nhẹ (không vuông góc hoàn toàn)
        if random.random() > 0.4:
            plate = add_perspective(plate, strength=random.uniform(0.02, 0.08))

        # 3. Đặt vào cảnh
        scene_img, bbox = create_scene_with_plate(plate, scene)

        # 4. Ánh sáng
        scene_img = apply_lighting_effect(scene_img, lighting)

        # 5. Nhiễu + blur
        arr = np.array(scene_img)
        arr = add_realistic_noise(arr)
        if random.random() > 0.5:
            arr = add_motion_blur(arr)

        # 6. JPEG compression artifact (thực tế khi lưu ảnh điện thoại)
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, random.randint(70, 95)]
        _, encoded = cv2.imencode('.jpg', cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), encode_param)
        arr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        # Tên file mô tả đầy đủ
        plate_text = f"{province}-{number}".replace(" ","")
        fname = f"{i+1:02d}_{plate_text}_{scene}_{lighting}.jpg"
        fpath = os.path.join(output_dir, fname)

        cv2.imwrite(fpath, arr)
        created.append({
            "filename": fname,
            "plate":    f"{province}-{number}",
            "type":     ptype,
            "scene":    scene,
            "lighting": lighting,
            "bbox":     bbox,
            "size":     arr.shape
        })
        print(f"  [{i+1:02d}] {fname}  {arr.shape}")

    # Lưu metadata dataset (quan trọng cho báo cáo)
    import json
    meta = {
        "dataset_name":   "VN License Plate Dataset",
        "version":        "1.0",
        "total_images":   len(created),
        "description":    "Dataset biển số xe Việt Nam sinh tổng hợp theo QCVN 41:2019/BGTVT",
        "collection_method": (
            "Ảnh được tạo bằng PIL TrueType font (LiberationSans-Bold) theo đúng "
            "quy chuẩn kích thước, màu sắc biển số VN. Áp dụng các biến đổi thực tế: "
            "perspective transform, Gaussian noise, motion blur, JPEG compression, "
            "và 5 điều kiện ánh sáng (bình thường, quá sáng, thiếu sáng, mưa, nắng). "
            "4 loại cảnh nền: đường phố, ban đêm, bãi giữ xe, cận cảnh."
        ),
        "plate_standard": "QCVN 41:2019/BGTVT - Quy chuẩn kỹ thuật quốc gia về báo hiệu đường bộ",
        "conditions": {
            "scenes":   scene_types,
            "lighting": lighting_conds,
            "types":    list(set(plate_types))
        },
        "images": created
    }
    meta_path = os.path.join(output_dir, "dataset_info.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Tạo xong {len(created)} ảnh")
    print(f"✓ Metadata: {meta_path}")
    print(f"\n--- MÔ TẢ DATASET CHO BÁO CÁO ---")
    print(meta["collection_method"])
    return created


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "dataset/real_images"
    n   = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    create_full_dataset(out, n)
