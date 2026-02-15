from ultralytics import YOLO
import cv2
import numpy as np
import time
import math

# ---------------- LOAD MODELS ----------------
model = YOLO("yolov8n.pt")          # traffic model
pothole_model = YOLO("best.pt")     # YOUR TRAINED POTHOLE MODEL

# ---------------- VIDEO PATHS ----------------
video_paths = [
    # add your vedios here 
     "videos/pot.mp4",       
    "videos/traffic.mp4",
    "videos/traffic2.mp4",
    "videos/traffic3.mp4"
]

caps = [cv2.VideoCapture(v) for v in video_paths]
lane_names = ["Lane 1", "Lane 2", "Lane 3", "Lane 4"]

# ---------------- SIGNAL SETTINGS ----------------
NORMAL_GREEN = 20
EXTRA_GREEN = 20
HEAVY_THRESHOLD = 15
LOW_DENSITY_THRESHOLD = 5

AMBULANCE_CLASSES = [5, 7, 2]   # bus, truck, car (temporary simulation)

STOP_DISTANCE_THRESHOLD = 5     # pixels
STOP_TIME_THRESHOLD = 16        # seconds

current_lane = 0
green_start_time = time.time()
green_time = NORMAL_GREEN

# ---------------- MODES ----------------
emergency_mode = False
ambulance_active_lane = None

accident_mode = False
accident_confirmed = False
accident_lane = None

pothole_mode = False   # 🕳️ manual control mode

# ---------------- TRACKING ----------------
previous_positions = [{} for _ in range(4)]
stop_start_time = [{} for _ in range(4)]

# ---------------- WINDOW FULLSCREEN ----------------
cv2.namedWindow("SMART TRAFFIC MANAGEMENT SYSTEM", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("SMART TRAFFIC MANAGEMENT SYSTEM", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ---------------- MAIN LOOP ----------------
while True:
    frames = []
    vehicle_counts = []
    ambulance_detected = [False] * 4
    pothole_detected = [False] * 4   # 🕳️ per lane

    # ---------- READ FRAMES ----------
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frames.append(frame)

    processed_frames = []

    # ---------- DETECTION ----------
    for i, frame in enumerate(frames):
        results = model(frame, verbose=False)[0]
        count = 0
        current_positions = {}

        # ---------------- VEHICLE + ACCIDENT ----------------
        for idx, box in enumerate(results.boxes):
            cls = int(box.cls)

            # Vehicle detection
            if cls in [1, 2, 3, 5, 7]:
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                current_positions[idx] = (cx, cy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # -------- AUTO ACCIDENT DETECTION --------
                if count <= LOW_DENSITY_THRESHOLD:
                    if idx in previous_positions[i]:
                        px, py = previous_positions[i][idx]
                        dist = math.hypot(cx - px, cy - py)

                        if dist < STOP_DISTANCE_THRESHOLD:
                            if idx not in stop_start_time[i]:
                                stop_start_time[i][idx] = time.time()
                            elif time.time() - stop_start_time[i][idx] >= STOP_TIME_THRESHOLD:
                                if not accident_mode:
                                    accident_mode = True
                                    accident_lane = i
                        else:
                            stop_start_time[i].pop(idx, None)

            # 🚑 Emergency detection
            if emergency_mode and cls in AMBULANCE_CLASSES:
                ambulance_detected[i] = True

        # ---------------- POTHOLE AI (MANUAL MODE) ----------------
        if pothole_mode:
            pothole_results = pothole_model(frame, verbose=False)[0]

            for box in pothole_results.boxes:
                cls = int(box.cls)
                conf = float(box.conf)

                # assuming class 0 = pothole
                if cls == 0 and conf > 0.5:
                    pothole_detected[i] = True

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 3)
                    cv2.putText(frame, "POTHOLE", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,140,255), 2)

        previous_positions[i] = current_positions
        vehicle_counts.append(count)
        processed_frames.append(frame)

    # ---------- ACCIDENT MODE (TOP PRIORITY) ----------
    if accident_mode:
        current_lane = accident_lane
        green_time = float("inf")

    # ---------- EMERGENCY MODE ----------
    elif emergency_mode:
        if ambulance_active_lane is None:
            for i in range(4):
                if ambulance_detected[i]:
                    ambulance_active_lane = i
                    break

        if ambulance_active_lane is not None:
            current_lane = ambulance_active_lane
            green_time = float("inf")

            if not ambulance_detected[ambulance_active_lane]:
                ambulance_active_lane = None
                emergency_mode = False
                green_start_time = time.time()
                green_time = NORMAL_GREEN
        else:
            emergency_mode = False
            green_start_time = time.time()
            green_time = NORMAL_GREEN

    # ---------- NORMAL MODE ----------
    else:
        elapsed = time.time() - green_start_time
        if elapsed >= green_time:
            current_lane = (current_lane + 1) % 4
            green_start_time = time.time()
            green_time = NORMAL_GREEN + EXTRA_GREEN if vehicle_counts[current_lane] > HEAVY_THRESHOLD else NORMAL_GREEN

    remaining_time = "∞" if green_time == float("inf") else max(0, int(green_time - (time.time() - green_start_time)))

    # ================= UI =================
    dashboard = []

    for i, frame in enumerate(processed_frames):
        frame = cv2.resize(frame, (640, 360))
        cv2.rectangle(frame, (0, 0), (640, 120), (25, 25, 25), -1)

        # Density
        if vehicle_counts[i] <= 5:
            density, d_color = "LOW", (0, 200, 0)
        elif vehicle_counts[i] <= 15:
            density, d_color = "MEDIUM", (0, 200, 200)
        else:
            density, d_color = "HIGH", (0, 0, 255)

        # Signal
        if i == current_lane:
            if accident_mode:
                signal, s_color, timer = "ACCIDENT CLEARANCE", (0, 255, 255), "∞"
            elif emergency_mode and ambulance_active_lane is not None:
                signal, s_color, timer = "EMERGENCY GREEN", (255, 0, 0), "∞"
            else:
                signal, s_color, timer = "GREEN", (0, 255, 0), f"{remaining_time}s"
        else:
            signal, s_color, timer = "RED", (0, 0, 255), ""

        cv2.putText(frame, lane_names[i], (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"Vehicles: {vehicle_counts[i]}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
        cv2.putText(frame, f"Density: {density}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, d_color, 2)
        cv2.putText(frame, f"{signal} {timer}", (360, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, s_color, 2)

        # pothole badge
        if pothole_mode and pothole_detected[i]:
            cv2.putText(frame, "POTHOLE ALERT", (360, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,140,255), 3)

        dashboard.append(frame)

    final_frame = np.vstack([
        np.hstack(dashboard[:2]),
        np.hstack(dashboard[2:])
    ])

    # ================= POPUPS =================
    if emergency_mode and ambulance_active_lane is not None:
        cv2.putText(final_frame,
                    "🚑 EMERGENCY VEHICLE DETECTED - GREEN CORRIDOR ACTIVE",
                    (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 3)

    if accident_mode and not accident_confirmed:
        cv2.putText(final_frame,
                    "⚠ AI DETECTED ACCIDENT - PRESS C TO CONFIRM",
                    (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 3)

    if accident_confirmed:
        cv2.putText(final_frame,
                    "🛠 ACCIDENT CONFIRMED - CLEARANCE IN PROGRESS",
                    (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 3)

    if pothole_mode and any(pothole_detected):
        cv2.putText(final_frame,
                    "🕳️ POTHOLE MODE ACTIVE - ROAD DAMAGE DETECTED",
                    (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 140, 255), 3)

    # Footer
    cv2.rectangle(final_frame, (0, final_frame.shape[0]-60), (final_frame.shape[1], final_frame.shape[0]), (20,20,20), -1)
    cv2.putText(final_frame,
                "E-Emergency  N-Normal  P-Pothole ON  K-Pothole OFF  C-Confirm Accident  R-Reset Accident  Q-Quit",
                (40, final_frame.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,180,180), 2)

    cv2.imshow("SMART TRAFFIC MANAGEMENT SYSTEM", final_frame)

    # ---------- KEYS ----------
    key = cv2.waitKey(25) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('e'):
        emergency_mode = True
        ambulance_active_lane = None
    elif key == ord('n'):
        emergency_mode = False
        ambulance_active_lane = None
        green_start_time = time.time()
        green_time = NORMAL_GREEN
    elif key == ord('p'):   # 🕳️ pothole ON
        pothole_mode = True
    elif key == ord('k'):   # 🕳️ pothole OFF
        pothole_mode = False
    elif key == ord('c') and accident_mode:
        accident_confirmed = True
    elif key == ord('r'):
        accident_mode = False
        accident_confirmed = False
        accident_lane = None
        green_start_time = time.time()
        green_time = NORMAL_GREEN

# ---------------- CLEANUP ----------------
for cap in caps:
    cap.release()
cv2.destroyAllWindows()

