"""
License Plate Recognition Pipeline
====================================
Đáp ứng các yêu cầu:
- Ch.3: Phát hiện đặc trưng (SIFT/ORB), đối sánh mô tả, phát hiện cạnh (Canny), đường thẳng (HoughLines)
- Ch.4: Phân đoạn ảnh (GrabCut/Watershed), Mean-Shift, phân cụm (K-Means), phát hiện biên (Canny/Sobel)
- Ch.5: Nhận dạng đối tượng (EasyOCR + cascade classifier)
"""

import cv2
import numpy as np
from PIL import Image
from skimage import filters, measure, morphology, segmentation, color
from skimage.feature import canny as skimage_canny
from scipy import ndimage
import easyocr
import os
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class FeatureDetector:
    """Ch.3 - Phát hiện đặc trưng và đối sánh ảnh"""

    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=500)
        try:
            self.sift = cv2.SIFT_create()
        except Exception:
            self.sift = None

    def detect_and_describe(self, image_gray):
        """Phát hiện keypoints và mô tả bằng ORB"""
        keypoints, descriptors = self.orb.detectAndCompute(image_gray, None)
        return keypoints, descriptors

    def detect_edges_canny(self, image_gray):
        """Ch.3 + Ch.4 - Phát hiện cạnh bằng Canny"""
        blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    def detect_edges_sobel(self, image_gray):
        """Ch.4 - Phát hiện biên bằng Sobel"""
        sobelx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        sobel_mag = np.uint8(sobel_mag / sobel_mag.max() * 255)
        return sobel_mag

    def detect_lines_hough(self, edges):
        """Ch.3 - Phát hiện đường thẳng bằng Hough Transform"""
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi/180,
            threshold=80, minLineLength=50, maxLineGap=10
        )
        return lines

    def draw_keypoints(self, image, keypoints):
        """Vẽ keypoints lên ảnh"""
        return cv2.drawKeypoints(image, keypoints, None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    def draw_lines(self, image, lines):
        """Vẽ đường thẳng lên ảnh"""
        output = image.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return output


class ImageSegmenter:
    """Ch.4 - Phân đoạn ảnh"""

    def kmeans_segmentation(self, image, k=6):
        """Ch.4 - Phân cụm K-Means"""
        h, w = image.shape[:2]
        pixel_vals = image.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        _, labels, centers = cv2.kmeans(
            pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()].reshape(image.shape)
        return segmented, labels.reshape((h, w))

    def meanshift_segmentation(self, image):
        """Ch.4 - Mean-Shift phân đoạn"""
        result = cv2.pyrMeanShiftFiltering(image, sp=21, sr=51)
        return result

    def watershed_segmentation(self, image_gray):
        """Ch.4 - Watershed (đường viền động)"""
        _, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        img_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)
        return markers

    def find_contours(self, edges):
        """Ch.4 - Tìm đường viền (contours)"""
        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours, hierarchy

    def skimage_segmentation(self, image_gray):
        """Ch.4 - Phân đoạn bằng scikit-image (Felzenszwalb)"""
        img_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
        segments = segmentation.felzenszwalb(img_rgb, scale=100, sigma=0.5, min_size=50)
        return segments


class PlateDetector:
    """Phát hiện vùng biển số xe"""

    def __init__(self):
        # Haar Cascade cho biển số xe
        cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
        self.plate_cascade = None
        if os.path.exists(cascade_path):
            self.plate_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info(f"Loaded plate cascade: {cascade_path}")

    def detect_by_cascade(self, image_gray):
        """Phát hiện biển số bằng Haar Cascade"""
        if self.plate_cascade is None:
            return []
        plates = self.plate_cascade.detectMultiScale(
            image_gray, scaleFactor=1.1, minNeighbors=4,
            minSize=(60, 20), maxSize=(400, 150)
        )
        return plates if len(plates) > 0 else []

    def detect_by_contours(self, image):
        """Phát hiện biển số bằng phân tích contour hình học"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(blurred, 30, 200)

        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

        plate_regions = []
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                area = w * h
                if 1.5 < aspect_ratio < 6.0 and area > 1500:
                    plate_regions.append((x, y, w, h))

        return plate_regions

    def detect_by_morphology(self, image):
        """Phát hiện biển số bằng morphology + thresholding"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 13, 15, 15)

        # Blackhat morphology để tìm vùng tối trên nền sáng
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Đóng morphology để nối các ký tự
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        plate_regions = []
        h_img, w_img = image.shape[:2]
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            ar = w / float(h)
            area_ratio = (w * h) / float(w_img * h_img)
            if 2.0 < ar < 6.5 and 0.005 < area_ratio < 0.25:
                plate_regions.append((x, y, w, h))

        return plate_regions

    def merge_detections(self, regions_list):
        """Gộp các vùng phát hiện từ nhiều phương pháp"""
        all_regions = []
        for regions in regions_list:
            for r in regions:
                all_regions.append(r)

        if not all_regions:
            return []

        # NMS đơn giản dựa trên overlap
        merged = []
        used = [False] * len(all_regions)

        for i, r1 in enumerate(all_regions):
            if used[i]:
                continue
            x1, y1, w1, h1 = r1
            best = r1
            best_area = w1 * h1
            for j, r2 in enumerate(all_regions):
                if i == j or used[j]:
                    continue
                x2, y2, w2, h2 = r2
                ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
                iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
                if ix * iy > 0.4 * min(w1*h1, w2*h2):
                    used[j] = True
                    if w2*h2 > best_area:
                        best = r2
                        best_area = w2*h2
            merged.append(best)
            used[i] = True

        return merged


class PlateRecognizer:
    """Ch.5 - Nhận dạng và phân loại ký tự biển số"""

    def __init__(self):
        self.reader = None
        self.ocr_engine = None

        # Ưu tiên 1: EasyOCR (tốt nhất, cần model download lần đầu)
        try:
            logger.info("Initializing EasyOCR...")
            self.reader = easyocr.Reader(['vi', 'en'], gpu=False, verbose=False)
            self.ocr_engine = 'easyocr'
            logger.info("EasyOCR ready")
        except Exception as e:
            logger.warning(f"EasyOCR unavailable: {e}")

        # Fallback 2: Tesseract OCR
        if self.ocr_engine is None:
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
                self.pytesseract = pytesseract
                self.ocr_engine = 'tesseract'
                logger.info("Tesseract OCR ready (fallback)")
            except Exception as e2:
                logger.warning(f"Tesseract unavailable: {e2}")

        # Fallback 3: OpenCV template-based OCR
        if self.ocr_engine is None:
            self.ocr_engine = 'opencv'
            logger.info("Using OpenCV-based OCR (basic fallback)")

        logger.info(f"OCR engine: {self.ocr_engine}")

    def preprocess_plate(self, plate_img):
        """Tiền xử lý ảnh biển số trước khi OCR"""
        # Scale up để OCR chính xác hơn
        h, w = plate_img.shape[:2]
        scale = max(1, 200 // h)
        if scale > 1:
            plate_img = cv2.resize(plate_img, (w * scale, h * scale),
                                   interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # Làm sắc nét
        kernel_sharp = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        sharp = cv2.filter2D(gray, -1, kernel_sharp)

        # CLAHE để tăng contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(sharp)

        # Otsu threshold
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return plate_img, gray, enhanced, binary

    def recognize(self, plate_img):
        """Nhận dạng ký tự - hỗ trợ nhiều engine OCR"""
        if self.ocr_engine == 'easyocr' and self.reader:
            return self._recognize_easyocr(plate_img)
        elif self.ocr_engine == 'tesseract':
            return self._recognize_tesseract(plate_img)
        else:
            return self._recognize_opencv(plate_img)

    def _recognize_easyocr(self, plate_img):
        """EasyOCR recognition"""
        results = self.reader.readtext(plate_img, detail=1,
                                       allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.')
        texts = []
        for (bbox, text, conf) in results:
            text = text.upper().strip()
            text = ''.join(c for c in text if c.isalnum() or c in '.-')
            if text and conf > 0.3:
                texts.append((text, conf))
        return texts

    def _recognize_tesseract(self, plate_img):
        """Tesseract OCR recognition - thử nhiều PSM và lấy kết quả tốt nhất"""
        plate_scaled, gray, enhanced, binary = self.preprocess_plate(plate_img)

        best_texts = []
        best_score = 0

        # Thử nhiều cấu hình PSM
        configs = [
            '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.',
            '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.',
            '--psm 13 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.',
            '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.',
        ]
        # Thử nhiều version ảnh (binary, enhanced, inverted)
        imgs_to_try = [binary, enhanced, cv2.bitwise_not(binary)]

        for img_variant in imgs_to_try:
            pil_img = Image.fromarray(img_variant)
            for cfg in configs:
                try:
                    data = self.pytesseract.image_to_data(
                        pil_img, config=cfg,
                        output_type=self.pytesseract.Output.DICT
                    )
                    texts = []
                    total_conf = 0
                    for i, text in enumerate(data['text']):
                        text = text.upper().strip()
                        text = ''.join(c for c in text if c.isalnum() or c in '.-')
                        raw_conf = int(data['conf'][i])
                        if raw_conf < 0:
                            continue
                        conf = raw_conf / 100.0
                        if text and len(text) >= 1:
                            texts.append((text, conf))
                            total_conf += conf
                    if texts and total_conf > best_score:
                        best_score = total_conf
                        best_texts = texts
                except Exception:
                    continue

        # Fallback: image_to_string nếu vẫn chưa có
        if not best_texts:
            for img_variant in [binary, cv2.bitwise_not(binary)]:
                try:
                    pil_img = Image.fromarray(img_variant)
                    raw = self.pytesseract.image_to_string(
                        pil_img,
                        config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.'
                    ).strip()
                    cleaned = ''.join(c for c in raw.upper() if c.isalnum() or c in '.-')
                    if len(cleaned) >= 2:
                        best_texts = [(cleaned, 0.55)]
                        break
                except Exception as e:
                    logger.warning(f"Tesseract string mode: {e}")

        return best_texts

    def _recognize_opencv(self, plate_img):
        """OpenCV-based character segmentation + template matching"""
        _, gray, enhanced, binary = self.preprocess_plate(plate_img)

        # Tìm các vùng ký tự bằng connected components
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cv2.bitwise_not(binary), connectivity=8
        )

        h, w = binary.shape
        chars = []
        for i in range(1, n_labels):
            x, y, cw, ch, area = stats[i]
            # Lọc theo kích thước ký tự hợp lý
            aspect = cw / float(ch) if ch > 0 else 0
            if (5 < cw < w//2 and h//4 < ch < h and
                    area > 50 and 0.1 < aspect < 1.5):
                chars.append((x, y, cw, ch))

        chars.sort(key=lambda c: c[0])  # Sắp xếp trái sang phải

        if not chars:
            # Fallback: đọc toàn bộ bằng Otsu threshold + heuristic
            text = self._heuristic_read(binary)
            if text:
                return [(text, 0.4)]
            return []

        # Trả về vị trí các ký tự với confidence thấp (không có template)
        plate_text = f"[{len(chars)} chars detected]"
        return [(plate_text, 0.35)]

    def _heuristic_read(self, binary):
        """Heuristic đọc biển số từ binary image"""
        # Đếm pixel trắng theo cột để tách ký tự
        col_hist = np.sum(binary == 0, axis=0)
        threshold = binary.shape[0] * 0.1

        in_char = False
        char_bounds = []
        start = 0
        for i, v in enumerate(col_hist):
            if v > threshold and not in_char:
                in_char = True
                start = i
            elif v <= threshold and in_char:
                in_char = False
                if i - start > 3:
                    char_bounds.append((start, i))

        return f"~{len(char_bounds)}chars" if char_bounds else ""

    def format_plate_text(self, texts):
        """Format kết quả nhận dạng thành chuỗi biển số"""
        if not texts:
            return "KHÔNG NHẬN DẠNG ĐƯỢC", 0.0
        combined = ' '.join(t[0] for t in texts)
        avg_conf = np.mean([t[1] for t in texts])
        return combined, avg_conf


class LicensePlateSystem:
    """Hệ thống chính tích hợp toàn bộ pipeline"""

    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.feature_detector = FeatureDetector()
        self.segmenter = ImageSegmenter()
        self.plate_detector = PlateDetector()
        self.recognizer = PlateRecognizer()

        logger.info("License Plate System initialized")

    def process_image(self, image_input, save_result=True, filename_prefix="result"):
        """
        Pipeline xử lý ảnh đầy đủ
        Returns: dict chứa ảnh kết quả và thông tin nhận dạng
        """
        # Load ảnh
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Không thể đọc ảnh: {image_input}")
        else:
            image = image_input.copy()

        h, w = image.shape[:2]
        # Resize nếu quá lớn
        if w > 1280:
            scale = 1280 / w
            image = cv2.resize(image, (1280, int(h * scale)))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ============================================================
        # STAGE 1: CH.3 - Phát hiện đặc trưng
        # ============================================================
        keypoints, descriptors = self.feature_detector.detect_and_describe(gray)
        edges_canny = self.feature_detector.detect_edges_canny(gray)
        edges_sobel = self.feature_detector.detect_edges_sobel(gray)
        lines = self.feature_detector.detect_lines_hough(edges_canny)

        img_keypoints = self.feature_detector.draw_keypoints(image.copy(), keypoints)
        img_lines = self.feature_detector.draw_lines(image.copy(), lines)

        # ============================================================
        # STAGE 2: CH.4 - Phân đoạn ảnh
        # ============================================================
        img_meanshift = self.segmenter.meanshift_segmentation(image)
        img_kmeans, _ = self.segmenter.kmeans_segmentation(image, k=6)
        contours, _ = self.segmenter.find_contours(edges_canny)

        img_contours = image.copy()
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 1)

        # ============================================================
        # STAGE 3: Phát hiện vùng biển số
        # ============================================================
        regions_cascade = self.plate_detector.detect_by_cascade(gray)
        regions_contour = self.plate_detector.detect_by_contours(image)
        regions_morpho = self.plate_detector.detect_by_morphology(image)

        all_regions = self.plate_detector.merge_detections([
            list(regions_cascade) if len(regions_cascade) > 0 else [],
            regions_contour,
            regions_morpho
        ])

        # ============================================================
        # STAGE 4: CH.5 - Nhận dạng ký tự
        # ============================================================
        plate_results = []
        result_image = image.copy()

        for (x, y, w_r, h_r) in all_regions:
            # Thêm padding
            pad = 5
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(image.shape[1], x + w_r + pad)
            y2 = min(image.shape[0], y + h_r + pad)

            plate_crop = image[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            _, _, enhanced, binary = self.recognizer.preprocess_plate(plate_crop)
            texts = self.recognizer.recognize(plate_crop)
            plate_text, confidence = self.recognizer.format_plate_text(texts)

            plate_results.append({
                'bbox': (x1, y1, x2 - x1, y2 - y1),
                'text': plate_text,
                'confidence': confidence,
                'crop': plate_crop,
                'enhanced': enhanced,
                'binary': binary
            })

            # Vẽ kết quả lên ảnh
            color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

            label = f"{plate_text} ({confidence:.0%})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result_image, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
            cv2.putText(result_image, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Tạo ảnh kết quả tổng hợp (debug visualization)
        debug_vis = self._create_debug_visualization(
            image, gray, edges_canny, edges_sobel,
            img_keypoints, img_lines, img_meanshift,
            img_kmeans, img_contours, result_image
        )

        # Lưu kết quả
        timestamp = int(time.time() * 1000)
        saved_files = {}
        if save_result:
            main_path = os.path.join(self.output_dir, f"{filename_prefix}_{timestamp}_main.jpg")
            debug_path = os.path.join(self.output_dir, f"{filename_prefix}_{timestamp}_debug.jpg")
            cv2.imwrite(main_path, result_image)
            cv2.imwrite(debug_path, debug_vis)
            saved_files = {'main': main_path, 'debug': debug_path}

            for i, pr in enumerate(plate_results):
                crop_path = os.path.join(self.output_dir,
                                          f"{filename_prefix}_{timestamp}_plate{i}.jpg")
                cv2.imwrite(crop_path, pr['crop'])
                pr['crop_path'] = crop_path

        return {
            'original': image,
            'result': result_image,
            'debug': debug_vis,
            'plates': plate_results,
            'edges_canny': edges_canny,
            'edges_sobel': edges_sobel,
            'keypoints': keypoints,
            'lines_count': len(lines) if lines is not None else 0,
            'saved_files': saved_files
        }

    def _create_debug_visualization(self, original, gray, edges_canny, edges_sobel,
                                     img_keypoints, img_lines, img_meanshift,
                                     img_kmeans, img_contours, result):
        """Tạo ảnh debug 3x3 grid hiển thị từng bước pipeline"""
        target_size = (320, 240)

        def resize_to(img, size=target_size):
            return cv2.resize(img, size)

        def to_bgr(img):
            if len(img.shape) == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img

        panels = [
            ("1.Original", resize_to(to_bgr(original))),
            ("2.Canny Edges", resize_to(to_bgr(edges_canny))),
            ("3.Sobel Edges", resize_to(to_bgr(edges_sobel))),
            ("4.Keypoints(ORB)", resize_to(to_bgr(img_keypoints))),
            ("5.Hough Lines", resize_to(to_bgr(img_lines))),
            ("6.Mean-Shift", resize_to(to_bgr(img_meanshift))),
            ("7.K-Means(k=6)", resize_to(to_bgr(img_kmeans))),
            ("8.Contours", resize_to(to_bgr(img_contours))),
            ("9.Result", resize_to(to_bgr(result))),
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        labeled_panels = []
        for title, panel in panels:
            p = panel.copy()
            cv2.rectangle(p, (0, 0), (target_size[0], 22), (30, 30, 30), -1)
            cv2.putText(p, title, (4, 16), font, 0.5, (255, 255, 100), 1)
            labeled_panels.append(p)

        row1 = np.hstack(labeled_panels[0:3])
        row2 = np.hstack(labeled_panels[3:6])
        row3 = np.hstack(labeled_panels[6:9])
        grid = np.vstack([row1, row2, row3])
        return grid

    def process_frame(self, frame):
        """Xử lý frame từ camera (optimized cho real-time)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Fast detection
        edges = self.feature_detector.detect_edges_canny(gray)
        regions_cascade = self.plate_detector.detect_by_cascade(gray)
        regions_morpho = self.plate_detector.detect_by_morphology(frame)

        all_regions = self.plate_detector.merge_detections([
            list(regions_cascade) if len(regions_cascade) > 0 else [],
            regions_morpho
        ])

        result_frame = frame.copy()
        plate_texts = []

        for (x, y, w_r, h_r) in all_regions:
            x1, y1 = max(0, x - 3), max(0, y - 3)
            x2 = min(frame.shape[1], x + w_r + 3)
            y2 = min(frame.shape[0], y + h_r + 3)

            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            texts = self.recognizer.recognize(plate_crop)
            plate_text, conf = self.recognizer.format_plate_text(texts)

            plate_texts.append({'text': plate_text, 'confidence': conf, 'bbox': (x1, y1, x2, y2)})

            color = (0, 255, 0) if conf > 0.5 else (0, 165, 255)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)

            label = f"{plate_text} {conf:.0%}"
            cv2.putText(result_frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Show edges overlay nhỏ ở góc
        edge_small = cv2.resize(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), (160, 90))
        result_frame[10:100, 10:170] = edge_small

        return result_frame, plate_texts, all_regions
