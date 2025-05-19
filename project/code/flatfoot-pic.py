import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO

# 載入 YOLOv8 模型
model = YOLO(r"C:/Users/user/Downloads/YOLOv8/ultralytics/segment/flatfoot/train5/weights/best.pt")  # 替換為你訓練好的模型路徑

# 自訂顏色
colors = {
    "Normalfoot": (0, 0, 255),
    "Flatfoot": (255, 0, 0)
}

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("足弓判斷")
        self.image_label = QLabel("請選擇圖片")
        self.result_label = QLabel("判斷結果: 尚未分析")
        self.result_label.setStyleSheet("font-size: 18pt; color: blue")

        self.button = QPushButton("選擇圖片")
        self.button.clicked.connect(self.load_image)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "選擇圖片", "", "Images (*.png *.xpm *.jpg *.jpeg)")
        if file_name:
            self.detect_image(file_name)

    def detect_image(self, image_path):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 推論
        results = model(img_rgb)[0]
        annotated = img_rgb.copy()

        if results.boxes is not None and len(results.boxes) > 0:
            # 找出信心度最高的 box
            best_idx = int(results.boxes.conf.argmax())
            box = results.boxes[best_idx]
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = box.conf[0]
            color = colors.get(label, (255, 255, 255))

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)
            cv2.putText(annotated, f"{label} {conf:.2f}", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            self.result_label.setText(f"判斷結果: {label}")
        else:
            self.result_label.setText("判斷結果: 無法辨識")

        # 顯示圖片
        height, width, channel = annotated.shape
        bytes_per_line = 3 * width
        qimage = QImage(annotated.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimage).scaled(640, 480, Qt.KeepAspectRatio))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())