"""
test_pipeline.py — Kiểm thử toàn bộ pipeline CV
Chạy: python utils/test_pipeline.py
"""
import sys, os, cv2, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pipeline import FeatureDetector, ImageSegmenter, PlateDetector, PlateRecognizer, LicensePlateSystem

results = {}

def test(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        results[name] = True
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        results[name] = False

def make_test_img():
    img = np.ones((480, 640, 3), dtype=np.uint8) * 100
    cv2.rectangle(img, (100, 150), (540, 380), (50, 80, 120), -1)
    cv2.rectangle(img, (200, 310), (440, 365), (255, 255, 255), -1)
    cv2.putText(img, "51A-123.45", (207, 353), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 180), 2)
    return cv2.add(img, np.random.randint(0, 20, img.shape, dtype=np.uint8))

print("\n" + "="*55)
print("  LICENSE PLATE CV PIPELINE — TEST SUITE")
print("="*55)

img  = make_test_img()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fd   = FeatureDetector()
seg  = ImageSegmenter()
pd_  = PlateDetector()

print("\n[Ch.3] Feature Detection & Edges & Lines")
def t_orb():
    kp, _ = fd.detect_and_describe(gray)
    assert len(kp) > 0
    print(f"         {len(kp)} keypoints")
test("ORB Keypoint Detection", t_orb)

def t_canny():
    e = fd.detect_edges_canny(gray)
    assert e.shape == gray.shape
    print(f"         shape={e.shape}")
test("Canny Edge Detection", t_canny)

def t_sobel():
    e = fd.detect_edges_sobel(gray)
    assert e.dtype == np.uint8
    print(f"         max={e.max()}")
test("Sobel Edge Detection", t_sobel)

def t_hough():
    edges = fd.detect_edges_canny(gray)
    lines = fd.detect_lines_hough(edges)
    print(f"         {len(lines) if lines is not None else 0} lines")
test("Hough Lines Transform", t_hough)

print("\n[Ch.4] Image Segmentation")
def t_kmeans():
    out, _ = seg.kmeans_segmentation(img, k=4)
    assert out.shape == img.shape
    print(f"         {out.shape}")
test("K-Means Clustering (k=4)", t_kmeans)

def t_ms():
    out = seg.meanshift_segmentation(img)
    assert out.shape == img.shape
    print(f"         {out.shape}")
test("Mean-Shift Filtering", t_ms)

def t_ws():
    m = seg.watershed_segmentation(gray)
    assert m.shape == gray.shape
    print(f"         {m.max()} regions")
test("Watershed Segmentation", t_ws)

def t_cnt():
    edges = fd.detect_edges_canny(gray)
    contours, _ = seg.find_contours(edges)
    print(f"         {len(contours)} contours")
test("Contour Detection", t_cnt)

def t_ski():
    s = seg.skimage_segmentation(gray)
    assert s.shape == gray.shape
    print(f"         {s.max()+1} segments")
test("scikit-image Felzenszwalb", t_ski)

print("\n[Plate Detection]")
def t_cas():
    r = pd_.detect_by_cascade(gray)
    print(f"         cascade: {len(list(r)) if hasattr(r,'__iter__') else 0} regions")
test("Haar Cascade", t_cas)

def t_geo():
    r = pd_.detect_by_contours(img)
    print(f"         contour: {len(r)} regions")
test("Geometric Contour", t_geo)

def t_mph():
    r = pd_.detect_by_morphology(img)
    print(f"         morpho: {len(r)} regions")
test("Morphological", t_mph)

def t_nms():
    r1 = pd_.detect_by_contours(img)
    r2 = pd_.detect_by_morphology(img)
    merged = pd_.merge_detections([r1, r2])
    print(f"         merged NMS: {len(merged)}")
test("NMS Fusion", t_nms)

print("\n[Ch.5] OCR Recognition")
rec = PlateRecognizer()

def t_pre():
    crop = img[300:370, 190:450]
    _, _, enhanced, binary = rec.preprocess_plate(crop)
    assert binary.dtype == np.uint8
    print(f"         engine={rec.ocr_engine}")
test("Plate Preprocessing", t_pre)

def t_ocr():
    crop = img[300:370, 190:450]
    texts = rec.recognize(crop)
    print(f"         {len(texts)} results: {[t[0] for t in texts[:3]]}")
test("OCR Recognition", t_ocr)

def t_fmt():
    text, conf = rec.format_plate_text([("51A", 0.9), ("123.45", 0.85)])
    assert "51A" in text
    print(f"         '{text}' conf={conf:.0%}")
test("Text Formatting", t_fmt)

print("\n[System] Full Integration Pipeline")
system = LicensePlateSystem(output_dir="results")

def t_full():
    t0 = time.time()
    r = system.process_image(img.copy(), save_result=True, filename_prefix="pytest")
    dt = time.time() - t0
    assert 'plates' in r and 'keypoints' in r
    assert r['edges_canny'] is not None and r['debug'] is not None
    assert r['saved_files']
    print(f"         {len(r['plates'])} plates | {len(r['keypoints'])} kp | {r['lines_count']} lines | {dt:.1f}s")
test("Full Pipeline + Save", t_full)

def t_dbg():
    r = system.process_image(img.copy(), save_result=False)
    h, w = r['debug'].shape[:2]
    print(f"         debug grid {w}x{h}px")
test("Debug 9-panel Visualization", t_dbg)

passed = sum(1 for v in results.values() if v)
total  = len(results)
print(f"\n{'='*55}")
print(f"  Result: {passed}/{total} passed")
if passed == total:
    print("  ALL TESTS PASSED!")
else:
    for k, v in results.items():
        if not v:
            print(f"  FAILED: {k}")
print("="*55 + "\n")
