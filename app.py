"""
╔══════════════════════════════════════════════════════════╗
║   Rock · Paper · Scissors  —  YOLO Detection UI         ║
║   Developed by Akshay Gurav                              ║
║                                                          ║
║   Press 'n'  → Switch Camera                            ║
║   Press ESC  → Quit                                     ║
╚══════════════════════════════════════════════════════════╝
"""

import cv2
import math
import time
import numpy as np
from ultralytics import YOLO

# ── Colour Palette (BGR) ────────────────────────────────
COLOURS = {
    "rock":     (0,  140, 255),   # vivid orange
    "paper":    (57, 255,  20),   # neon green
    "scissors": (255,  50, 100),  # hot pink/magenta
    "unknown":  (200, 200, 200),  # grey fallback
}

# UI accent colours (BGR)
UI_BG        = (15,  15,  20)    # near-black panel bg
UI_ACCENT    = (0,  200, 255)    # cyan accent
UI_TEXT      = (240, 240, 240)   # off-white text
UI_DIM       = (120, 120, 130)   # dim grey


# ── Helper: draw a semi-transparent filled rectangle ────
def filled_rect(img, pt1, pt2, colour, alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, colour, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


# ── Helper: rounded rectangle (no built-in in OpenCV) ───
def rounded_rect(img, pt1, pt2, colour, radius=14, thickness=2):
    x1, y1 = pt1
    x2, y2 = pt2
    r = radius
    cv2.line(img,  (x1+r, y1),   (x2-r, y1),   colour, thickness)
    cv2.line(img,  (x1+r, y2),   (x2-r, y2),   colour, thickness)
    cv2.line(img,  (x1,   y1+r), (x1,   y2-r), colour, thickness)
    cv2.line(img,  (x2,   y1+r), (x2,   y2-r), colour, thickness)
    cv2.ellipse(img, (x1+r, y1+r), (r, r), 180, 0, 90, colour, thickness)
    cv2.ellipse(img, (x2-r, y1+r), (r, r), 270, 0, 90, colour, thickness)
    cv2.ellipse(img, (x1+r, y2-r), (r, r),  90, 0, 90, colour, thickness)
    cv2.ellipse(img, (x2-r, y2-r), (r, r),   0, 0, 90, colour, thickness)


# ── Helper: draw detection box with label pill ──────────
def draw_detection(frame, x1, y1, x2, y2, label, conf, colour):
    # Outer glow (slightly expanded box, dark translucent)
    glow_expand = 6
    filled_rect(frame,
                (x1 - glow_expand, y1 - glow_expand),
                (x2 + glow_expand, y2 + glow_expand),
                colour, alpha=0.15)

    # Main bounding box (rounded)
    rounded_rect(frame, (x1, y1), (x2, y2), colour, radius=12, thickness=2)

    # Corner accents
    corner_len = 18
    corner_thickness = 3
    corners = [
        ((x1, y1 + corner_len), (x1, y1), (x1 + corner_len, y1)),
        ((x2 - corner_len, y1), (x2, y1), (x2, y1 + corner_len)),
        ((x1, y2 - corner_len), (x1, y2), (x1 + corner_len, y2)),
        ((x2 - corner_len, y2), (x2, y2), (x2, y2 - corner_len)),
    ]
    for (pa, pb, pc) in corners:
        cv2.line(frame, pa, pb, colour, corner_thickness)
        cv2.line(frame, pb, pc, colour, corner_thickness)

    # Label pill background
    pill_text  = f"  {label.upper()}  {int(conf*100)}%  "
    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.55
    thickness  = 1
    (tw, th), baseline = cv2.getTextSize(pill_text, font, font_scale, thickness)

    pill_y1 = max(y1 - th - 18, 4)
    pill_y2 = max(y1 - 4, th + 8)
    pill_x1 = x1
    pill_x2 = x1 + tw + 4

    # Pill fill
    filled_rect(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), colour, alpha=0.85)
    rounded_rect(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), colour, radius=6, thickness=1)

    # Pill text (dark for contrast)
    cv2.putText(frame, pill_text,
                (pill_x1 + 2, pill_y2 - baseline - 2),
                font, font_scale, (10, 10, 10), thickness, cv2.LINE_AA)


# ── Helper: draw HUD overlay ────────────────────────────
def draw_hud(frame, cam_idx, fps, detections):
    h, w = frame.shape[:2]

    # ── Top bar ──────────────────────────────────────────
    bar_h = 48
    filled_rect(frame, (0, 0), (w, bar_h), UI_BG, alpha=0.80)
    cv2.line(frame, (0, bar_h), (w, bar_h), UI_ACCENT, 1)

    # Title
    cv2.putText(frame, "RPS DETECTOR",
                (16, 32), cv2.FONT_HERSHEY_DUPLEX, 0.85,
                UI_ACCENT, 1, cv2.LINE_AA)

    # FPS badge
    fps_str = f"FPS {fps:05.1f}"
    cv2.putText(frame, fps_str,
                (w - 140, 32), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                UI_TEXT, 1, cv2.LINE_AA)

    # Camera badge
    cam_str = f"CAM {cam_idx}"
    cv2.putText(frame, cam_str,
                (w - 260, 32), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                UI_DIM, 1, cv2.LINE_AA)

    # ── Bottom bar ───────────────────────────────────────
    bot_h = 42
    filled_rect(frame, (0, h - bot_h), (w, h), UI_BG, alpha=0.80)
    cv2.line(frame, (0, h - bot_h), (w, h - bot_h), UI_ACCENT, 1)

    # Detected classes summary
    if detections:
        counts = {}
        for cls_name, _ in detections:
            counts[cls_name] = counts.get(cls_name, 0) + 1

        x_cursor = 16
        for cls_name, count in counts.items():
            colour = COLOURS.get(cls_name, COLOURS["unknown"])
            tag = f"● {cls_name.upper()} x{count}"
            cv2.putText(frame, tag,
                        (x_cursor, h - 14),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55,
                        colour, 1, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            x_cursor += tw + 24
    else:
        cv2.putText(frame, "No detections",
                    (16, h - 14), cv2.FONT_HERSHEY_DUPLEX, 0.55,
                    UI_DIM, 1, cv2.LINE_AA)

    # Key hints (right side of bottom bar)
    hints = "[N] Switch Cam    [ESC] Quit"
    cv2.putText(frame, hints,
                (w - 310, h - 14), cv2.FONT_HERSHEY_DUPLEX, 0.45,
                UI_DIM, 1, cv2.LINE_AA)


# ── Load YOLO model ─────────────────────────────────────
print("[INFO] Loading YOLO model (model.pt) ...")
try:
    model = YOLO("model.pt")
    classNames = ["rock", "paper", "scissors"]
    print("[OK]  Model loaded successfully.\n")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit(1)


# ── Camera auto-detect ──────────────────────────────────
def get_camera(start_index=0):
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
    for i in range(start_index, 6):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[OK]  Camera found at index {i}")
                return cap, i
            cap.release()
    return None, -1


cap, current_index = get_camera(0)
if cap is None:
    print("[ERROR] No camera found.")
    exit(1)

print("="*52)
print("  RPS Detector — Premium UI  |  by Akshay Gurav")
print("="*52)
print("  'n'  → Switch to next camera")
print("  ESC  → Quit")
print("="*52 + "\n")

# ── FPS tracking ─────────────────────────────────────────
fps        = 0.0
prev_time  = time.time()
fail_count = 0

# ── Main loop ────────────────────────────────────────────
while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        fail_count += 1
        if fail_count > 30:
            print("[WARN] Camera stalled. Try pressing 'n' to switch.")
            fail_count = 0
        continue
    fail_count = 0

    # FPS calculation
    now      = time.time()
    fps      = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
    prev_time = now

    # ── YOLO inference ──────────────────────────────────
    results    = model(frame, stream=True, verbose=False)
    detections = []   # list of (class_name, confidence)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            conf  = round(float(box.conf[0]), 2)
            cls   = int(box.cls[0])
            name  = classNames[cls] if cls < len(classNames) else "unknown"
            colour = COLOURS.get(name, COLOURS["unknown"])

            detections.append((name, conf))
            draw_detection(frame, x1, y1, x2, y2, name, conf, colour)

    # ── HUD overlay ────────────────────────────────────
    draw_hud(frame, current_index, fps, detections)

    cv2.imshow("RPS Detector  |  Akshay Gurav", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:   # ESC
        break
    elif key == ord('n'):
        print("\n[INFO] Switching camera ...")
        cap.release()
        cap, current_index = get_camera(current_index + 1)
        if cap is None:
            print("[INFO] Wrapping back to index 0.")
            cap, current_index = get_camera(0)
        if cap is None:
            print("[ERROR] No cameras found!")
            break

cap.release()
cv2.destroyAllWindows()
print("\n[DONE] Detector closed. Bye!")
