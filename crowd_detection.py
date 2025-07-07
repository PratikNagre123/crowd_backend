import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
import time
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os

app = Flask(__name__)
CORS(app)

model = YOLO("yolov8n.pt")

PROXIMITY_THRESHOLD = 250
CROWD_THRESHOLD = 6
ALERT_INTERVAL = 2
last_sound_time = 0
crowd_alert_active = False

def detect_objects(frame):
    results = model(frame)
    person_bboxes = []

    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            if int(cls) == 0:
                person_bboxes.append((int(x1), int(y1), int(x2), int(y2)))
    return person_bboxes

def play_alert_sound():
    print("[ALERT] Crowd detected!")

def check_crowding(bboxes):
    global last_sound_time, crowd_alert_active

    if len(bboxes) < CROWD_THRESHOLD:
        crowd_alert_active = False
        return False

    centers = np.array([((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2 in bboxes])
    adaptive_proximity = min(PROXIMITY_THRESHOLD + len(bboxes) * 2, 200)

    clustering = DBSCAN(eps=adaptive_proximity, min_samples=3, metric='euclidean').fit(centers)
    cluster_labels = clustering.labels_
    unique_labels, counts = np.unique(cluster_labels[cluster_labels != -1], return_counts=True)

    if any(count >= CROWD_THRESHOLD for count in counts):
        current_time = time.time()
        if current_time - last_sound_time > ALERT_INTERVAL:
            last_sound_time = current_time
            threading.Thread(target=play_alert_sound, daemon=True).start()
        crowd_alert_active = True
        return True

    crowd_alert_active = False
    return False

def run_crowd_analysis(video_path, selected_flags):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return "[ERROR] Could not open video file."

    frame_count = 0
    detected_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        bboxes = detect_objects(frame)
        if check_crowding(bboxes):
            timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            detected_frames.append(f"[ALERT] Crowd detected at {timestamp:02d} sec")

    cap.release()
    return "\n".join(detected_frames) if detected_frames else "[INFO] No crowd detected."

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    flags_str = request.form.get('flags', '[]')
    selected_flags = json.loads(flags_str)

    temp_video_path = f"temp_{time.time()}.mp4"
    video_file.save(temp_video_path)

    try:
        analysis_log = run_crowd_analysis(temp_video_path, selected_flags)
    except Exception as e:
        analysis_log = f"[ERROR] Exception occurred: {str(e)}"

    os.remove(temp_video_path)

    return jsonify({'log': analysis_log})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
