import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os
import time

# 🔥 RESET CSV EVERY RUN
if os.path.exists("smart_crowd_results.csv"):
    os.remove("smart_crowd_results.csv")

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("dataset_video.mp4")  # or use 0 for webcam

if not os.path.exists("static"):
    os.makedirs("static")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame_count += 1

    frame = cv2.resize(frame, (640, 360))

    results = model(frame)

    persons = []

    # ✅ DETECT PEOPLE
    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                persons.append(center)

    # ✅ LEFT / RIGHT
    left_count = 0
    right_count = 0

    for (x, y) in persons:
        if x < frame.shape[1] // 2:
            left_count += 1
        else:
            right_count += 1

    total_people = len(persons)

    # ✅ SAVE DATA EVERY FRAME (IMPORTANT)
    df = pd.DataFrame([[
        frame_count,
        total_people,
        left_count,
        right_count
    ]], columns=["Frame", "Crowd Count", "Left Zone", "Right Zone"])

    if os.path.exists("smart_crowd_results.csv"):
        df.to_csv("smart_crowd_results.csv", mode='a', header=False, index=False)
    else:
        df.to_csv("smart_crowd_results.csv", index=False)

    # 🔥 HEATMAP
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    for (x, y) in persons:
        cv2.circle(heatmap, (x, y), 25, 1, -1)

    heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

    cv2.imwrite("static/heatmap.jpg", overlay)

    # 🔥 SLOW DOWN (IMPORTANT for UI update)
    time.sleep(0.5)

cap.release()
cv2.destroyAllWindows()