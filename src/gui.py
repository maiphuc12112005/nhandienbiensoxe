"""
GUI Application - License Plate Recognition System
Giao diện đồ họa người dùng sử dụng Tkinter
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import os
import sys
import time
import queue
import json
from datetime import datetime

# Thêm đường dẫn src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from pipeline import LicensePlateSystem
except ImportError:
    from src.pipeline import LicensePlateSystem


class LicensePlateGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚗 Hệ Thống Nhận Diện Biển Số Xe")
        self.root.geometry("1280x800")
        self.root.configure(bg="#0f172a")
        self.root.minsize(1000, 700)

        self.system = None
        self.camera_active = False
        self.camera_thread = None
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.current_image = None
        self.processing = False
        self.captured_plates = []
        self.output_dir = "results"
        os.makedirs(self.output_dir, exist_ok=True)

        self._setup_styles()
        self._build_ui()
        self._init_system_async()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('Title.TLabel',
                        background='#0f172a', foreground='#38bdf8',
                        font=('Courier New', 11, 'bold'))

        style.configure('Info.TLabel',
                        background='#1e293b', foreground='#94a3b8',
                        font=('Courier New', 9))

        style.configure('Success.TLabel',
                        background='#1e293b', foreground='#4ade80',
                        font=('Courier New', 10, 'bold'))

        style.configure('Dark.TFrame', background='#0f172a')
        style.configure('Card.TFrame', background='#1e293b')
        style.configure('TNotebook', background='#0f172a', borderwidth=0)
        style.configure('TNotebook.Tab',
                        background='#1e293b', foreground='#94a3b8',
                        font=('Courier New', 9, 'bold'), padding=[12, 6])
        style.map('TNotebook.Tab',
                  background=[('selected', '#0ea5e9')],
                  foreground=[('selected', 'white')])

        style.configure('Accent.TButton',
                        background='#0ea5e9', foreground='white',
                        font=('Courier New', 9, 'bold'), borderwidth=0, padding=8)
        style.map('Accent.TButton',
                  background=[('active', '#38bdf8'), ('disabled', '#334155')])

        style.configure('Danger.TButton',
                        background='#ef4444', foreground='white',
                        font=('Courier New', 9, 'bold'), borderwidth=0, padding=8)
        style.map('Danger.TButton',
                  background=[('active', '#f87171')])

        style.configure('Success.TButton',
                        background='#22c55e', foreground='white',
                        font=('Courier New', 9, 'bold'), borderwidth=0, padding=8)
        style.map('Success.TButton',
                  background=[('active', '#4ade80')])

        style.configure('Prog.Horizontal.TProgressbar',
                        background='#0ea5e9', troughcolor='#1e293b',
                        borderwidth=0, lightcolor='#0ea5e9', darkcolor='#0ea5e9')

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg='#020617', height=60)
        header.pack(fill='x')
        header.pack_propagate(False)

        tk.Label(header, text="◈ LICENSE PLATE RECOGNITION SYSTEM",
                 font=('Courier New', 16, 'bold'),
                 bg='#020617', fg='#38bdf8').pack(side='left', padx=20, pady=10)

        tk.Label(header, text="CV Pipeline: Feature Detection | Segmentation | Recognition",
                 font=('Courier New', 9),
                 bg='#020617', fg='#475569').pack(side='left', padx=10, pady=10)

        self.status_label = tk.Label(header, text="⟳ Đang khởi tạo...",
                                      font=('Courier New', 9, 'bold'),
                                      bg='#020617', fg='#f59e0b')
        self.status_label.pack(side='right', padx=20)

        # Main layout
        main = tk.Frame(self.root, bg='#0f172a')
        main.pack(fill='both', expand=True, padx=10, pady=5)

        # Left panel - controls
        left = tk.Frame(main, bg='#1e293b', width=260)
        left.pack(side='left', fill='y', padx=(0, 8), pady=0)
        left.pack_propagate(False)
        self._build_left_panel(left)

        # Center - display
        center = tk.Frame(main, bg='#0f172a')
        center.pack(side='left', fill='both', expand=True)
        self._build_center_panel(center)

        # Right panel - results
        right = tk.Frame(main, bg='#1e293b', width=280)
        right.pack(side='right', fill='y', padx=(8, 0))
        right.pack_propagate(False)
        self._build_right_panel(right)

        # Status bar
        statusbar = tk.Frame(self.root, bg='#020617', height=24)
        statusbar.pack(fill='x', side='bottom')
        statusbar.pack_propagate(False)

        self.statusbar_label = tk.Label(
            statusbar,
            text="© License Plate Recognition System | CV Pipeline",
            font=('Courier New', 8), bg='#020617', fg='#334155'
        )
        self.statusbar_label.pack(side='left', padx=10)

        self.fps_label = tk.Label(statusbar, text="FPS: --",
                                   font=('Courier New', 8, 'bold'),
                                   bg='#020617', fg='#0ea5e9')
        self.fps_label.pack(side='right', padx=10)

    def _build_left_panel(self, parent):
        tk.Label(parent, text="◆ ĐIỀU KHIỂN",
                 font=('Courier New', 10, 'bold'),
                 bg='#1e293b', fg='#38bdf8').pack(pady=(15, 5), padx=10, anchor='w')

        tk.Frame(parent, bg='#334155', height=1).pack(fill='x', padx=10, pady=5)

        # Image section
        tk.Label(parent, text="[ PHÂN TÍCH ẢNH ]",
                 font=('Courier New', 8, 'bold'),
                 bg='#1e293b', fg='#64748b').pack(padx=10, anchor='w', pady=(10, 3))

        ttk.Button(parent, text="📁  Mở Ảnh",
                   style='Accent.TButton',
                   command=self._open_image).pack(fill='x', padx=10, pady=3)

        ttk.Button(parent, text="⚡  Phân Tích",
                   style='Success.TButton',
                   command=self._analyze_image).pack(fill='x', padx=10, pady=3)

        tk.Frame(parent, bg='#334155', height=1).pack(fill='x', padx=10, pady=10)

        # Camera section
        tk.Label(parent, text="[ CAMERA REAL-TIME ]",
                 font=('Courier New', 8, 'bold'),
                 bg='#1e293b', fg='#64748b').pack(padx=10, anchor='w', pady=(0, 3))

        cam_frame = tk.Frame(parent, bg='#1e293b')
        cam_frame.pack(fill='x', padx=10, pady=3)

        tk.Label(cam_frame, text="Camera ID:",
                 font=('Courier New', 8),
                 bg='#1e293b', fg='#94a3b8').pack(side='left')

        self.cam_id_var = tk.StringVar(value="0")
        cam_entry = tk.Entry(cam_frame, textvariable=self.cam_id_var,
                             width=4, font=('Courier New', 9),
                             bg='#0f172a', fg='#e2e8f0',
                             insertbackground='white', relief='flat',
                             highlightthickness=1, highlightcolor='#0ea5e9',
                             highlightbackground='#334155')
        cam_entry.pack(side='right')

        self.cam_btn = ttk.Button(parent, text="▶  Bật Camera",
                                   style='Accent.TButton',
                                   command=self._toggle_camera)
        self.cam_btn.pack(fill='x', padx=10, pady=3)

        ttk.Button(parent, text="📸  Chụp & Lưu",
                   style='Success.TButton',
                   command=self._capture_frame).pack(fill='x', padx=10, pady=3)

        tk.Frame(parent, bg='#334155', height=1).pack(fill='x', padx=10, pady=10)

        # View options
        tk.Label(parent, text="[ CHẾ ĐỘ HIỂN THỊ ]",
                 font=('Courier New', 8, 'bold'),
                 bg='#1e293b', fg='#64748b').pack(padx=10, anchor='w', pady=(0, 3))

        self.view_mode = tk.StringVar(value="result")
        modes = [
            ("Kết quả chính", "result"),
            ("Debug pipeline", "debug"),
            ("Canny edges", "canny"),
            ("Sobel edges", "sobel"),
        ]
        for text, val in modes:
            tk.Radiobutton(parent, text=text, variable=self.view_mode,
                           value=val, command=self._update_display,
                           font=('Courier New', 8),
                           bg='#1e293b', fg='#94a3b8',
                           selectcolor='#0f172a',
                           activebackground='#1e293b',
                           activeforeground='#38bdf8').pack(padx=15, anchor='w')

        tk.Frame(parent, bg='#334155', height=1).pack(fill='x', padx=10, pady=10)

        # Stats
        tk.Label(parent, text="[ THỐNG KÊ ]",
                 font=('Courier New', 8, 'bold'),
                 bg='#1e293b', fg='#64748b').pack(padx=10, anchor='w', pady=(0, 5))

        stats_frame = tk.Frame(parent, bg='#0f172a')
        stats_frame.pack(fill='x', padx=10, pady=3)

        self.stat_frames = {}
        stats = [
            ("Đã xử lý", "processed", "0"),
            ("Tìm thấy", "found", "0"),
            ("Lưu ảnh", "saved", "0"),
        ]
        for label, key, val in stats:
            row = tk.Frame(stats_frame, bg='#0f172a')
            row.pack(fill='x', pady=2)
            tk.Label(row, text=f"  {label}:", font=('Courier New', 8),
                     bg='#0f172a', fg='#64748b').pack(side='left')
            lbl = tk.Label(row, text=val, font=('Courier New', 9, 'bold'),
                           bg='#0f172a', fg='#4ade80')
            lbl.pack(side='right', padx=5)
            self.stat_frames[key] = lbl

        tk.Frame(parent, bg='#334155', height=1).pack(fill='x', padx=10, pady=10)

        ttk.Button(parent, text="🗑  Xóa Kết Quả",
                   style='Danger.TButton',
                   command=self._clear_results).pack(fill='x', padx=10, pady=3)

        ttk.Button(parent, text="📂  Mở Thư Mục",
                   style='Accent.TButton',
                   command=self._open_output_dir).pack(fill='x', padx=10, pady=3)

        # Progress
        tk.Frame(parent, bg='#334155', height=1).pack(fill='x', padx=10, pady=10)
        self.progress = ttk.Progressbar(parent, style='Prog.Horizontal.TProgressbar',
                                         mode='indeterminate')
        self.progress.pack(fill='x', padx=10, pady=5)

    def _build_center_panel(self, parent):
        # Image canvas
        canvas_frame = tk.Frame(parent, bg='#0f172a')
        canvas_frame.pack(fill='both', expand=True)

        # Main display
        self.canvas = tk.Canvas(canvas_frame, bg='#020617',
                                 highlightthickness=1,
                                 highlightbackground='#1e293b',
                                 cursor='crosshair')
        self.canvas.pack(fill='both', expand=True)

        # Placeholder text
        self.canvas.create_text(
            400, 300,
            text="◈ Mở ảnh hoặc bật camera để bắt đầu",
            font=('Courier New', 14), fill='#334155',
            tags='placeholder'
        )
        self.canvas.create_text(
            400, 330,
            text="Hỗ trợ: JPG, PNG, BMP | Camera: USB/Webcam",
            font=('Courier New', 9), fill='#1e293b',
            tags='placeholder'
        )

        # Info bar below canvas
        info_bar = tk.Frame(parent, bg='#1e293b', height=30)
        info_bar.pack(fill='x')
        info_bar.pack_propagate(False)

        self.img_info_label = tk.Label(
            info_bar, text="Không có ảnh",
            font=('Courier New', 8), bg='#1e293b', fg='#475569'
        )
        self.img_info_label.pack(side='left', padx=10)

        self.detect_info_label = tk.Label(
            info_bar, text="",
            font=('Courier New', 8, 'bold'), bg='#1e293b', fg='#4ade80'
        )
        self.detect_info_label.pack(side='right', padx=10)

    def _build_right_panel(self, parent):
        tk.Label(parent, text="◆ KẾT QUẢ NHẬN DẠNG",
                 font=('Courier New', 10, 'bold'),
                 bg='#1e293b', fg='#38bdf8').pack(pady=(15, 5), padx=10, anchor='w')

        tk.Frame(parent, bg='#334155', height=1).pack(fill='x', padx=10, pady=5)

        # Plate display area
        self.plate_frame = tk.Frame(parent, bg='#1e293b')
        self.plate_frame.pack(fill='x', padx=10, pady=5)

        self.plate_canvas = tk.Canvas(self.plate_frame, bg='#0f172a',
                                       height=80, highlightthickness=0)
        self.plate_canvas.pack(fill='x')
        self.plate_canvas.create_text(
            130, 40, text="Biển số sẽ hiện ở đây",
            font=('Courier New', 9), fill='#334155'
        )

        # Big text display
        self.plate_text_var = tk.StringVar(value="---")
        plate_display = tk.Frame(parent, bg='#020617')
        plate_display.pack(fill='x', padx=10, pady=5)

        self.plate_big_label = tk.Label(
            plate_display,
            textvariable=self.plate_text_var,
            font=('Courier New', 22, 'bold'),
            bg='#020617', fg='#f8fafc',
            pady=10
        )
        self.plate_big_label.pack()

        self.conf_label = tk.Label(
            plate_display,
            text="Độ tin cậy: --",
            font=('Courier New', 9),
            bg='#020617', fg='#64748b'
        )
        self.conf_label.pack()

        tk.Frame(parent, bg='#334155', height=1).pack(fill='x', padx=10, pady=8)

        # Log
        tk.Label(parent, text="[ NHẬT KÝ ]",
                 font=('Courier New', 8, 'bold'),
                 bg='#1e293b', fg='#64748b').pack(padx=10, anchor='w', pady=(0, 3))

        self.log_text = scrolledtext.ScrolledText(
            parent, height=12, width=30,
            font=('Courier New', 8),
            bg='#020617', fg='#94a3b8',
            insertbackground='white',
            relief='flat',
            state='disabled'
        )
        self.log_text.pack(fill='both', expand=True, padx=10, pady=5)

        # History
        tk.Frame(parent, bg='#334155', height=1).pack(fill='x', padx=10, pady=5)

        tk.Label(parent, text="[ LỊCH SỬ ]",
                 font=('Courier New', 8, 'bold'),
                 bg='#1e293b', fg='#64748b').pack(padx=10, anchor='w')

        self.history_listbox = tk.Listbox(
            parent, height=6,
            font=('Courier New', 8),
            bg='#020617', fg='#4ade80',
            selectbackground='#0ea5e9',
            relief='flat', borderwidth=0,
            activestyle='none'
        )
        self.history_listbox.pack(fill='x', padx=10, pady=5)

        ttk.Button(parent, text="💾  Xuất Log",
                   style='Accent.TButton',
                   command=self._export_log).pack(fill='x', padx=10, pady=5)

    def _init_system_async(self):
        """Khởi tạo system trong thread riêng để không block GUI"""
        self.progress.start()

        def init():
            try:
                self.system = LicensePlateSystem(output_dir=self.output_dir)
                self.root.after(0, self._on_system_ready)
            except Exception as e:
                self.root.after(0, lambda: self._on_system_error(str(e)))

        threading.Thread(target=init, daemon=True).start()

    def _on_system_ready(self):
        self.progress.stop()
        self.status_label.config(text="✓ Sẵn sàng", fg='#4ade80')
        self._log("✓ Hệ thống khởi tạo thành công")
        self._log("→ Hỗ trợ: ORB/SIFT features, Canny/Sobel edges")
        self._log("→ K-Means, Mean-Shift, Watershed segmentation")
        self._log("→ EasyOCR + Cascade plate detection")

    def _on_system_error(self, err):
        self.progress.stop()
        self.status_label.config(text="✗ Lỗi khởi tạo", fg='#ef4444')
        self._log(f"✗ Lỗi: {err}")
        messagebox.showerror("Lỗi", f"Không thể khởi tạo hệ thống:\n{err}")

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.config(state='normal')
        self.log_text.insert('end', f"[{ts}] {msg}\n")
        self.log_text.see('end')
        self.log_text.config(state='disabled')

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh biển số",
            filetypes=[
                ("Ảnh", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("Tất cả", "*.*")
            ]
        )
        if path:
            self.current_image_path = path
            img = cv2.imread(path)
            if img is not None:
                self.current_image = img
                self._display_cv_image(img)
                h, w = img.shape[:2]
                self.img_info_label.config(
                    text=f"Ảnh: {os.path.basename(path)} | {w}×{h}px"
                )
                self._log(f"→ Mở ảnh: {os.path.basename(path)}")
                self.canvas.delete('placeholder')

    def _analyze_image(self):
        if self.current_image is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng mở ảnh trước!")
            return
        if self.system is None:
            messagebox.showwarning("Cảnh báo", "Hệ thống chưa sẵn sàng!")
            return
        if self.processing:
            return

        self.processing = True
        self.progress.start()
        self.status_label.config(text="⟳ Đang xử lý...", fg='#f59e0b')
        self._log("→ Bắt đầu phân tích pipeline...")

        def analyze():
            try:
                prefix = "img_" + os.path.splitext(
                    os.path.basename(self.current_image_path))[0]
                result = self.system.process_image(
                    self.current_image.copy(),
                    save_result=True,
                    filename_prefix=prefix
                )
                self.root.after(0, lambda: self._on_analysis_done(result))
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.root.after(0, lambda: self._on_analysis_error(str(e), tb))

        threading.Thread(target=analyze, daemon=True).start()

    def _on_analysis_done(self, result):
        self.processing = False
        self.progress.stop()
        self.last_result = result

        plates = result['plates']
        n_plates = len(plates)
        n_lines = result['lines_count']
        n_kp = len(result['keypoints'])

        self.status_label.config(text="✓ Hoàn tất", fg='#4ade80')
        self._log(f"✓ Phân tích xong: {n_plates} biển số")
        self._log(f"  → Keypoints: {n_kp} | Hough lines: {n_lines}")

        self._update_stats(n_plates)
        self._update_display()

        if plates:
            best = max(plates, key=lambda p: p['confidence'])
            self.plate_text_var.set(best['text'])
            self.conf_label.config(
                text=f"Độ tin cậy: {best['confidence']:.1%}",
                fg='#4ade80' if best['confidence'] > 0.6 else '#f59e0b'
            )
            self._show_plate_crop(best['crop'])

            # Add to history
            ts = datetime.now().strftime("%H:%M")
            entry = f"[{ts}] {best['text']} ({best['confidence']:.0%})"
            self.history_listbox.insert(0, entry)
            self.captured_plates.append({
                'time': datetime.now().isoformat(),
                'plate': best['text'],
                'confidence': best['confidence']
            })

            self._log(f"  → Biển số: {best['text']} ({best['confidence']:.1%})")
        else:
            self.plate_text_var.set("KHÔNG TÌM THẤY")
            self.conf_label.config(text="Thử ảnh khác hoặc điều chỉnh độ sáng",
                                   fg='#ef4444')
            self._log("  → Không phát hiện biển số")

        self.detect_info_label.config(
            text=f"✓ {n_plates} biển số | {n_kp} keypoints | {n_lines} lines"
        )

        # Cập nhật stat saved
        saved = len([f for f in os.listdir(self.output_dir)
                     if f.endswith(('_main.jpg', '_plate0.jpg'))])
        self.stat_frames['saved'].config(text=str(saved))

    def _on_analysis_error(self, err, tb=""):
        self.processing = False
        self.progress.stop()
        self.status_label.config(text="✗ Lỗi", fg='#ef4444')
        self._log(f"✗ Lỗi xử lý: {err}")
        messagebox.showerror("Lỗi xử lý", f"{err}\n\n{tb[:300]}")

    def _update_display(self):
        if not hasattr(self, 'last_result'):
            return
        result = self.last_result
        mode = self.view_mode.get()

        if mode == "result":
            img = result['result']
        elif mode == "debug":
            img = result['debug']
        elif mode == "canny":
            img = cv2.cvtColor(result['edges_canny'], cv2.COLOR_GRAY2BGR)
        elif mode == "sobel":
            img = cv2.cvtColor(result['edges_sobel'], cv2.COLOR_GRAY2BGR)
        else:
            img = result['result']

        self._display_cv_image(img)

    def _display_cv_image(self, cv_img):
        """Hiển thị ảnh OpenCV lên canvas"""
        self.canvas.delete('placeholder')
        canvas_w = self.canvas.winfo_width() or 800
        canvas_h = self.canvas.winfo_height() or 550

        h, w = cv_img.shape[:2]
        scale = min(canvas_w / w, canvas_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)

        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).resize((new_w, new_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil_img)

        self.canvas.delete('all')
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2
        self.canvas.create_image(x, y, anchor='nw', image=photo, tags='img')
        self.canvas._img_ref = photo  # Giữ reference

    def _show_plate_crop(self, crop_img):
        """Hiển thị ảnh biển số crop"""
        if crop_img is None or crop_img.size == 0:
            return
        h, w = crop_img.shape[:2]
        scale = min(260 / w, 80 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(crop_img, (new_w, new_h))
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        photo = ImageTk.PhotoImage(pil_img)

        self.plate_canvas.delete('all')
        x = (260 - new_w) // 2
        y = (80 - new_h) // 2
        self.plate_canvas.create_image(x, y, anchor='nw', image=photo)
        self.plate_canvas._img_ref = photo

    def _toggle_camera(self):
        if not self.camera_active:
            self._start_camera()
        else:
            self._stop_camera()

    def _start_camera(self):
        if self.system is None:
            messagebox.showwarning("Cảnh báo", "Hệ thống chưa sẵn sàng!")
            return

        try:
            cam_id = int(self.cam_id_var.get())
        except ValueError:
            cam_id = 0

        self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", f"Không thể mở camera {cam_id}")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.camera_active = True
        self.cam_btn.config(text="■  Tắt Camera", style='Danger.TButton')
        self.status_label.config(text="● LIVE", fg='#ef4444')
        self._log(f"→ Bật camera {cam_id}")

        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()
        self._update_camera_display()

    def _stop_camera(self):
        self.camera_active = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.cam_btn.config(text="▶  Bật Camera", style='Accent.TButton')
        self.status_label.config(text="✓ Camera đã tắt", fg='#94a3b8')
        self._log("→ Tắt camera")

    def _camera_loop(self):
        """Loop đọc frame từ camera"""
        prev_time = time.time()
        process_interval = 0.5  # Xử lý OCR mỗi 0.5s
        last_process = 0

        while self.camera_active and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            now = time.time()
            fps = 1.0 / (now - prev_time) if now > prev_time else 0
            prev_time = now

            # Xử lý nhận diện định kỳ
            if now - last_process >= process_interval:
                try:
                    result_frame, plate_texts, regions = self.system.process_frame(frame)
                    last_process = now
                except Exception:
                    result_frame = frame.copy()
                    plate_texts = []
                    regions = []
            else:
                # Chỉ vẽ bounding box mà không OCR lại
                result_frame = frame.copy()
                plate_texts = []

            # FPS overlay
            cv2.putText(result_frame, f"FPS: {fps:.1f}",
                        (result_frame.shape[1] - 100, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Đẩy vào queue
            try:
                self.frame_queue.put_nowait((result_frame, plate_texts, fps))
            except queue.Full:
                pass

    def _update_camera_display(self):
        """Cập nhật hiển thị camera (chạy trong main thread)"""
        if not self.camera_active:
            return

        try:
            frame, plate_texts, fps = self.frame_queue.get_nowait()
            self._display_cv_image(frame)
            self.fps_label.config(text=f"FPS: {fps:.1f}")

            if plate_texts:
                best = max(plate_texts, key=lambda p: p['confidence'])
                self.plate_text_var.set(best['text'])
                self.conf_label.config(
                    text=f"Độ tin cậy: {best['confidence']:.1%}",
                    fg='#4ade80' if best['confidence'] > 0.5 else '#f59e0b'
                )
        except queue.Empty:
            pass

        self.root.after(33, self._update_camera_display)  # ~30fps refresh

    def _capture_frame(self):
        """Chụp và lưu frame hiện tại"""
        if not self.camera_active or self.cap is None:
            messagebox.showwarning("Cảnh báo", "Camera chưa bật!")
            return

        ret, frame = self.cap.read()
        if ret:
            ts = int(time.time())
            path = os.path.join(self.output_dir, f"capture_{ts}.jpg")
            cv2.imwrite(path, frame)
            self._log(f"✓ Lưu frame: capture_{ts}.jpg")

            n = int(self.stat_frames['saved'].cget('text') or 0)
            self.stat_frames['saved'].config(text=str(n + 1))

            # Phân tích frame đã chụp
            self.current_image = frame
            self.current_image_path = path
            if self.system:
                threading.Thread(
                    target=lambda: self.root.after(
                        0, self._analyze_image
                    ), daemon=True
                ).start()

    def _update_stats(self, n_found):
        n = int(self.stat_frames['processed'].cget('text') or 0)
        self.stat_frames['processed'].config(text=str(n + 1))

        total_found = int(self.stat_frames['found'].cget('text') or 0)
        self.stat_frames['found'].config(text=str(total_found + n_found))

    def _clear_results(self):
        if messagebox.askyesno("Xác nhận", "Xóa tất cả kết quả?"):
            self.history_listbox.delete(0, 'end')
            self.captured_plates.clear()
            self.plate_text_var.set("---")
            self.conf_label.config(text="Độ tin cậy: --", fg='#64748b')
            for key in self.stat_frames:
                self.stat_frames[key].config(text="0")
            self._log("✓ Đã xóa lịch sử")

    def _open_output_dir(self):
        import subprocess, platform
        if platform.system() == 'Windows':
            os.startfile(self.output_dir)
        elif platform.system() == 'Darwin':
            subprocess.run(['open', self.output_dir])
        else:
            subprocess.run(['xdg-open', self.output_dir])

    def _export_log(self):
        if not self.captured_plates:
            messagebox.showinfo("Thông báo", "Chưa có kết quả để xuất!")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("Tất cả", "*.*")],
            initialfile=f"log_{int(time.time())}.json"
        )
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.captured_plates, f, ensure_ascii=False, indent=2)
            self._log(f"✓ Xuất log: {os.path.basename(path)}")

    def on_close(self):
        self._stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = LicensePlateGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
