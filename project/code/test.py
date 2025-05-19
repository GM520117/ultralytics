import cv2

cap = cv2.VideoCapture("http://192.168.50.87:8080/video")

while True:
    ret, frame = cap.read()
    if not ret:
        print("無法讀取影像")
        break
    cv2.imshow("IP Cam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()