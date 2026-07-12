
# Traffic Detection & Speed Estimation

A computer vision pipeline that detects, tracks, and estimates the speed of vehicles from CCTV footage using YOLOv8 and OpenCV.

## Features
- Vehicle detection and tracking (cars, motorcycles, buses, trucks) using YOLOv8
- Virtual tripwire-based speed estimation
- Vehicle counting
- CSV export of vehicle speed data with timestamps
- Annotated output video with bounding boxes and speed overlay

## How it works
1. Two virtual tripwires are drawn across the video frame
2. When a tracked vehicle crosses the first line, its entry frame is recorded
3. When it crosses the second line, elapsed time is calculated and used to estimate speed (distance ÷ time)
4. Results are logged to `traffic_data.csv` and drawn onto the output video

## Files
- `traffic_output.avi` — annotated output video
- `traffic_data.csv` — logged vehicle speed data

## Tech stack
- YOLOv8 (Ultralytics)
- OpenCV
- ByteTrack
