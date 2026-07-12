import cv2
import csv
import time
from ultralytics import YOLO

# 1. Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# 2. Open the input video file
video_path = '/content/traffic_data/video/cctv052x2004080618x00080.avi'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = '/content/traffic_output.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

csv_filename = '/content/traffic_data.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Vehicle_ID", "Speed_kmh"])

TRIPWIRE_1_Y = int(height*0.2)
TRIPWIRE_2_Y = int(height*0.8)

REAL_WORLD_DISTANCE_METERS = 10

entry_frames = {}
counted_ids = set()
total_vehicles = 0
frame_count = 0

print("Processing video... Please wait. This won't display live output.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    results = model.track(frame, classes=[2, 3, 5, 7], persist=True, tracker="/content/my_bytetrack.yaml", verbose=False)

    cv2.line(frame, (0, TRIPWIRE_1_Y), (frame.shape[1], TRIPWIRE_1_Y), (0, 255, 255), 1)
    cv2.line(frame, (0, TRIPWIRE_2_Y), (frame.shape[1], TRIPWIRE_2_Y), (0, 0, 255), 1)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if track_id in counted_ids:
               box_color = (0, 0, 255)
            else:
                box_color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            if cy > TRIPWIRE_1_Y and cy < TRIPWIRE_2_Y and track_id not in entry_frames:
                entry_frames[track_id] = frame_count

            if cy > TRIPWIRE_2_Y and track_id not in counted_ids:
                counted_ids.add(track_id)
                total_vehicles += 1

                if track_id in entry_frames:
                    frames_elapsed = frame_count - entry_frames[track_id]

                    if frames_elapsed > 0:
                        time_seconds = frames_elapsed / fps
                        speed_mps = REAL_WORLD_DISTANCE_METERS / time_seconds
                        speed_kmh = int(speed_mps * 3.6)

                        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                        with open(csv_filename, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([current_time, track_id, speed_kmh])

                        cv2.putText(frame, f"{speed_kmh} km/h", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.putText(frame, f"Total Vehicles Counted: {total_vehicles}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    out.write(frame)

cap.release()
out.release()

print(f"Video processing complete! Total vehicles: {total_vehicles}")
print(f"Your video is saved at: {output_path}")
print(f"Data saved at: {csv_filename}")
