import cv2

# DroidCamX 串流 URL
droidcam_url = "http://192.168.171.116:4747/mjpegfeed"

# 嘗試開啟串流
cap = cv2.VideoCapture(droidcam_url)

# 檢查是否成功開啟串流
if not cap.isOpened():
    print("❌ 無法開啟串流，請檢查 DroidCamX 是否正在運行！")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print("❌ 無法獲取影像，請檢查串流是否正常！")
        break

    # 顯示影像
    cv2.imshow("DroidCamX Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
