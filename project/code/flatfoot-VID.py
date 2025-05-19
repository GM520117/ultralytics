import sys
import cv2
import numpy as np
import contextlib
import os
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from ultralytics import YOLO

# 模型載入（可改成 yolov8n-seg.pt 等輕量模型）
model = YOLO(r"C:/Users/user/Downloads/YOLOv8/ultralytics/segment//flatfoottrain5/weights/best.pt")

# 不顯示 YOLO 推論 log
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            yield

# 顏色對應
colors = {
    "Normalfoot": (0, 0, 255),
    "Flatfoot": (255, 0, 0)
}

class YOLOThread(QThread):
    result_ready = pyqtSignal(np.ndarray, str)  # 傳回影像與標籤

    def __init__(self, camera_link):
        super().__init__()
        self.camera_link = camera_link
        self.cap = cv2.VideoCapture(self.camera_link)
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with suppress_stdout():
                results = model(img_rgb)[0]

            annotated = img_rgb.copy()
            label_text = "判斷結果: 無法辨識"
            found = False

            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    conf = box.conf[0]
                    if conf < 0.6:
                        continue  # 跳過信心度太低的框

                    found = True
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    color = colors.get(label, (255, 255, 255))

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)
                    cv2.putText(annotated, f"{label} {conf:.2f}", (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                if found:
                    label_text = f"判斷結果: {label}"

            self.result_ready.emit(annotated, label_text)

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.quit()
        self.wait()

# 📺 主視窗
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("足弓即時判斷")
        self.image_label = QLabel()
        self.result_label = QLabel("判斷結果: 尚未分析")
        self.result_label.setStyleSheet("font-size: 18pt; color: blue")

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

        # IP 相機連結
        ip_address = "10.22.54.143"
        port = "8080"
        camera_link = f"http://{ip_address}:{port}/video"

        self.yolo_thread = YOLOThread(camera_link)
        self.yolo_thread.result_ready.connect(self.update_gui)
        self.yolo_thread.start()

    def update_gui(self, frame: np.ndarray, label_text: str):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimage = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimage).scaled(640, 480, Qt.KeepAspectRatio))
        self.result_label.setText(label_text)

    def closeEvent(self, event):
        self.yolo_thread.stop()

# 🚀 啟動
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())