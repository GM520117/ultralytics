import cv2
import time
import numpy as np
import urllib.request

from ultralytics import YOLO

# Load YOLOv8 segmentation model
model_path = r"C:/Users/owner/Downloads/YOLOv8/ultralytics/segment/train1/weights/best.pt"
model = YOLO(model_path)

# Set DroidCamX video streaming URL
#droidcam_url = "http://192.168.171.116:4747/video"
droidcam_url = "http://192.168.171.116:4747/mjpegfeed"
#camera = cv2.VideoCapture(droidcam_url)

# Set the color corresponding to the category
class_colors = {
    "Intact Pill": (255, 0, 0),  # Blue
    "Chipped Pill": (0, 0, 255)  # Red
}

def get_frame():
    try:
        img_resp = urllib.request.urlopen(droidcam_url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)
        return frame
    except Exception as e:
        print(f"Error: Failed to retrieve image. {e}")
        return None

# Timestamp used to calculate FPS
prev_time = time.time()
frame_count = 0

while True:
    frame = get_frame()
    if frame is None:
        print("Error: Failed to capture image.")
        continue  # Try to get the image again
    
    # Get the original image size
    h, w, _ = frame.shape

    # Calculate scaling and keep proportions
    scale = min(640 / w, 640 / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Adjust image size
    resized_frame = cv2.resize(frame, (new_w, new_h))

    # Create a black filled background
    padded_frame = np.zeros((640, 40, 3), dtype=np.uint8)
    y_offset = (640 - new_h) // 2
    x_offset = (640 - new_w) // 2

    # Put the adjusted image into a black background
    padded_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame

    # Execute YOLOv8 Segmentation inference
    results = model(padded_frame)

    # Draw detection results
    for result in results:
        for box, conf, cls, mask in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls, result.masks.xy):
            x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
            class_name = model.names[int(cls)]
            color = class_colors.get(class_name, (0, 255, 0))  #Default green
            label = f"{class_name} {conf:.2f}"

            # Draw Segmentation Mask
            mask_pts = np.array(mask, dtype=np.int32)
            cv2.fillPoly(frame, [mask_pts], color)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # mark text
            cv2.putText(padded_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    frame_count += 1
    if frame_count >= 10:
        curr_time = time.time()
        fps = frame_count / (curr_time - prev_time)
        prev_time = curr_time
        frame_count = 0 

    cv2.imshow("YOLOv8 Segmentation Detection", padded_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()