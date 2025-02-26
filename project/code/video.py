import cv2
import os

from ultralytics import YOLO

# 載入已訓練的模型 (使用預訓練模型或自訂模型)
model_path = r"C:/Users/owner/Downloads/YOLOv8/ultralytics/segment/train1/weights/best.pt"
model = YOLO(model_path)

# 讀取影片
video_path = r"C:/Users/owner/Downloads/YOLOv8/ultralytics/project/videos/v1.mp4"  # 替換為你的影片檔案
cap = cv2.VideoCapture(video_path)

# 取得影片資訊
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 設定輸出影片
output_floder = r"C:/Users/owner/Downloads/YOLOv8/ultralytics/project/results_VD"
os.makedirs(output_floder, exist_ok=True)  # 確保資料夾存在
output_path = os.path.join(output_floder, "output_video.mp4")
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 結束條件

    # 進行物件偵測
    results = model(frame)

    # 在影像上畫出偵測結果
    annotated_frame = results[0].plot()

    # 顯示影像 (可選)
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    
    # 儲存影片
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()
