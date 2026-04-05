import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from scipy.spatial import distance
import os

model = YOLO("yolov8n.pt")

CROWD_THRESHOLD = 4
DISTANCE_THRESHOLD = 60

crowd_data = []

cap = cv2.VideoCapture("dataset_video.mp4")

if not os.path.exists("static"):
    os.makedirs("static")

frame_count = 0
MAX_FRAMES = 15   # slightly increased

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count > MAX_FRAMES:
        break

    frame = cv2.resize(frame, (640, 360))

    results = model(frame)

    persons = []

    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                persons.append(center)

    # 🔥 SMART CROWD LOGIC
    crowd_count = 0
    for i, p1 in enumerate(persons):
        close = 0
        for j, p2 in enumerate(persons):
            if i != j and distance.euclidean(p1, p2) < DISTANCE_THRESHOLD:
                close += 1
        if close >= CROWD_THRESHOLD:
            crowd_count += 1

    crowd_data.append([frame_count, crowd_count])

    # 🔥 ADVANCED HEATMAP
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    for (x, y) in persons:
        cv2.circle(heatmap, (x, y), 50, 1, -1)

    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

    cv2.imwrite("static/heatmap.jpg", overlay)

# SAVE CSV
df = pd.DataFrame(crowd_data, columns=["Frame", "Crowd Count"])
df.to_csv("smart_crowd_results.csv", index=False)

cap.release()
cv2.destroyAllWindows()