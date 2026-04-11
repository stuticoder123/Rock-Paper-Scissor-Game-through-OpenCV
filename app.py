"""
OpenCV Camera - YOLO Object Detection with Camera Switcher
Developed by Akshay Gurav

Press 'n' to switch to next camera if you are seeing a virtual camera.
Press ESC to quit.
"""

import cv2
import math
from ultralytics import YOLO

# ── Load YOLO Model ─────────────────────────────────────
print("[INFO] Loading YOLO model (model.pt)...")
try:
    model = YOLO("model.pt")
    classNames = ["rock", "paper", "scissors"]
    print("[OK] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit(1)

# ── Auto-detect camera ──────────────────────────────────
def get_camera(start_index=0):
    # Try finding a working camera from start_index to 5
    for i in range(start_index, 5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[OK] Camera found at index {i}")
                return cap, i
            cap.release()
    return None, -1

cap, current_index = get_camera(0)
if cap is None:
    print("[ERROR] No camera found.")
    exit(1)

print("\n" + "="*50)
print("[OK] Camera open!")
print("👉 Press 'n' to switch to the NEXT camera.")
print("   (Use this if you see 'Animaze' or a Virtual Camera instead of your face)")
print("👉 Press ESC to quit.")
print("="*50 + "\n")

# ── Main loop ───────────────────────────────────────────
while True:
    ret, frame = cap.read()
    
    if not ret or frame is None:
        continue

    # === YOLO INFERENCE ===
    # Using verbose=False to stop printing prediction details on every frame
    results = model(frame, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence & class
            confidence = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            label = f"{classNames[cls]} {confidence:.2f}"

            # put text (dynamically placing text above the box)
            org = (x1, max(25, y1 - 10))
            cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # ======================

    # Show the frame with detections
    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:   # ESC
        break
    elif key == ord('n'):
        print("\n[INFO] Switching camera... (Searching for next available camera)")
        cap.release()
        
        # Search for the next id
        cap, current_index = get_camera(current_index + 1)
        
        # If no more cameras are available, loop back to 0
        if cap is None:
            print("[INFO] Reached end of cameras, looping back to index 0.")
            cap, current_index = get_camera(0)
            
        if cap is None:
            print("[ERROR] No cameras found at all!")
            break

cap.release()
cv2.destroyAllWindows()
print("[DONE] Camera closed.")
