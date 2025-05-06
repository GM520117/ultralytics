from ultralytics import YOLO

model = YOLO(r"C:\Users\user\Downloads\YOLOv8\ultralytics\segment\train-FCB(100-640)\weights\best.pt")  # 你的模型
metrics = model.val(data=r"C:\Users\user\Downloads\YOLOv8\ultralytics\Capsule.v9-fcb-two.yolov8\data.yaml")  # 輸出驗證結果

# metrics 會自動包含 precision、recall、F1-score、mAP 等
print(metrics.box.f1)  # 或 metrics.box.precision, etc.
