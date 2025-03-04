import cv2
import time
import numpy as np
import urllib.request

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from ultralytics import YOLO

# 設定 Selenium，開啟瀏覽器並典籍 Override 連結
def setup_droidcam():
    # 啟動 WebDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    driver.get("http://10.22.54.143:4747")  # 替換為你的 DroidCamX 網址

    time.sleep(2)  # 等待頁面加載

    try:
        override_link = driver.find_element(By.XPATH, "//a[@href='/override']")
        override_link.click()
        print("✅ 成功點擊 Override 連結")
        time.sleep(2)  # 等待影像載入
    except Exception as e:
        print(f"❌ 無法點擊 Override 連結: {e}")

    return driver  # 保持瀏覽器開啟

# 設定 YOLOv8 模型
model_path = r"C:/Users/owner/Downloads/YOLOv8/ultralytics/segment/train1/weights/best.pt"
model = YOLO(model_path)

# 設定 DroidCamX 影片串流 URL
droidcam_url = "http://10.22.54.143:4747/video?960x720"  # 預設影像流
#droidcam_url = "http://10.22.54.143:4747/mjpegfeed"  # MJPEG 格式
#droidcam_url = "http://10.22.54.143:4747/shot.jpg"   # 單張圖片模式

# 設定分類顏色
class_colors = {
    "Intact Pill": (255, 0, 0),  # Blue
    "Chipped Pill": (0, 0, 255)  # Red
}

def get_frame():
    try:
        # 嘗試不同的影像來源
        img_resp = urllib.request.urlopen(droidcam_url)  # 嘗試預設的 /video
        img_data = img_resp.read()

        if len(img_data) < 100:  # 確保影像大小合理
            raise ValueError("Received empty image data")

        img_np = np.array(bytearray(img_data), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)

        if frame is None:
            raise ValueError("cv2.imdecode() returned None")

        return frame
    except Exception as e:
        print(f"❌ 影像擷取失敗: {e}")
        return None

# 啟動 Selenium 來點擊超連結
browser = setup_droidcam()

# 計算 FPS
prev_time = time.time()
frame_count = 0

# 開始即時影像辨識
while True:
    frame = get_frame()
    if frame is None:
        print("Error: Failed to capture image.")
        continue  # 再次嘗試擷取影像
    
    # 取得原始影像尺寸
    h, w, _ = frame.shape

    # 計算縮放比例，保持 640x640
    scale = min(640 / w, 640 / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # 調整影像大小
    resized_frame = cv2.resize(frame, (new_w, new_h))

    # 建立黑色背景
    padded_frame = np.zeros((640, 40, 3), dtype=np.uint8)
    y_offset = (640 - new_h) // 2
    x_offset = (640 - new_w) // 2

    # 將影像置入黑色背景
    padded_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame

    # YOLOv8 進行辨識
    results = model(padded_frame)

    # 繪製結果
    for result in results:
        for box, conf, cls, mask in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls, result.masks.xy):
            x1, y1, x2, y2 = map(int, box)  # 邊界框
            class_name = model.names[int(cls)]
            color = class_colors.get(class_name, (0, 255, 0))  # 預設綠色
            label = f"{class_name} {conf:.2f}"

            # 繪製 Segmentation Mask
            mask_pts = np.array(mask, dtype=np.int32)
            cv2.fillPoly(frame, [mask_pts], color)

            #繪製邊界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 紀錄文字
            cv2.putText(padded_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    frame_count += 1
    if frame_count >= 10:
        curr_time = time.time()
        fps = frame_count / (curr_time - prev_time)
        prev_time = curr_time
        frame_count = 0 

    cv2.imshow("YOLOv8 Segmentation Detection", padded_frame)

    # 按下 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 結束城市，關閉視窗與瀏覽器
cv2.destroyAllWindows()
browser.quit()  # 關閉 Selenium