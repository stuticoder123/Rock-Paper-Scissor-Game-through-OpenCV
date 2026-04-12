"""
╔═══════════════════════════════════════════════════════════════════════╗
║          ROCK · PAPER · SCISSORS  ──  PRO EDITION                     ║
║          Developed by Stuti Gupta                                      ║
║                                                                       ║
║  SPLIT-SCREEN LAYOUT:                                                 ║
║   LEFT  → AI panel  (shows AI move as animated gesture art)           ║
║   RIGHT → Your camera feed with YOLO detection overlay                ║
║   CENTER→ VS badge, score bar, round timer                            ║
║                                                                       ║
║  HOW TO PLAY:                                                         ║
║   1. Show your hand gesture to the camera                             ║
║   2. Press SPACE to lock in your move (1s countdown)                  ║
║   3. AI reveals its move  →  Winner announced!                        ║
║   4. Play until all rounds done                                       ║
║                                                                       ║
║  CONTROLS:                                                            ║
║   SPACE  → Lock detected move                                         ║
║   R      → Restart / New game                                         ║
║   N      → Switch camera                                              ║
║   ESC    → Quit                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

import cv2
import time
import math
import random
import numpy as np
from ultralytics import YOLO

# ────────────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE  (all BGR)
# ────────────────────────────────────────────────────────────────────────────
C = {
    # gesture accent colours
    "rock":     (0,   130, 255),   # vivid orange
    "paper":    (30,  220,  80),   # emerald green
    "scissors": (220,  50, 180),   # hot magenta

    # UI colours
    "bg_dark":  ( 8,   8,  12),    # near-black
    "bg_mid":   (18,  18,  26),    # dark panel
    "bg_panel": (24,  24,  36),    # lighter panel
    "accent":   (0,  210, 255),    # cyan
    "gold":     (40, 200, 255),    # golden-white
    "white":    (240, 240, 245),
    "dim":      (100, 100, 115),
    "win":      ( 20, 220, 100),   # green
    "lose":     ( 50,  60, 220),   # red-blue
    "draw":     (200, 190,  40),   # yellow
    "divider":  ( 40,  40,  55),
}

FONT      = cv2.FONT_HERSHEY_DUPLEX
FONT_BOLD = cv2.FONT_HERSHEY_TRIPLEX

# Gesture display names
GESTURE_LABEL = {"rock": "ROCK", "paper": "PAPER", "scissors": "SCISSORS"}

# Load Real Images
GESTURE_IMAGES = {}
try:
    for g in ["rock", "paper", "scissors"]:
        img = cv2.imread(f"images/{g}.png")
        if img is not None:
            GESTURE_IMAGES[g] = img
except:
    pass

# ────────────────────────────────────────────────────────────────────────────
#  DRAWING PRIMITIVES
# ────────────────────────────────────────────────────────────────────────────

def alpha_rect(img, pt1, pt2, colour, alpha=0.6):
    ov = img.copy()
    cv2.rectangle(ov, pt1, pt2, colour, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def round_rect(img, pt1, pt2, colour, r=12, t=2):
    x1, y1 = pt1; x2, y2 = pt2
    r = min(r, (x2-x1)//2, (y2-y1)//2)
    for (ax,ay),(bx,by),(ang) in [
        ((x1+r,y1),(x1+r,y1+r),180),
        ((x2-r,y1),(x2-r,y1+r),270),
        ((x1+r,y2),(x1+r,y2-r), 90),
        ((x2-r,y2),(x2-r,y2-r),  0),
    ]:
        cv2.ellipse(img,(bx,by),(r,r),ang,0,90,colour,t)
    cv2.line(img,(x1+r,y1),(x2-r,y1),colour,t)
    cv2.line(img,(x1+r,y2),(x2-r,y2),colour,t)
    cv2.line(img,(x1,y1+r),(x1,y2-r),colour,t)
    cv2.line(img,(x2,y1+r),(x2,y2-r),colour,t)


def text_size(text, fs, t=1):
    (w, h), bl = cv2.getTextSize(text, FONT, fs, t)
    return w, h, bl


def put_centered(img, text, cx, cy, fs=0.7, colour=None, bold=False, t=1):
    colour = colour or C["white"]
    f = FONT_BOLD if bold else FONT
    (tw, th), _ = cv2.getTextSize(text, f, fs, t)
    cv2.putText(img, text, (cx - tw//2, cy + th//2), f, fs, colour, t, cv2.LINE_AA)


def put_left(img, text, x, cy, fs=0.6, colour=None, t=1):
    colour = colour or C["white"]
    (tw, th), _ = cv2.getTextSize(text, FONT, fs, t)
    cv2.putText(img, text, (x, cy + th//2), FONT, fs, colour, t, cv2.LINE_AA)


# ────────────────────────────────────────────────────────────────────────────
#  GESTURE ART  –  pure OpenCV vector drawings
# ────────────────────────────────────────────────────────────────────────────

def draw_gesture_art(canvas, cx, cy, gesture, colour, scale=1.0, pulse=0.0):
    """Draw a stylised hand gesture icon centred at (cx,cy)."""
    s = scale
    # Pulse: gentle size oscillation
    pulse_s = s * (1.0 + 0.04 * math.sin(pulse * 6.28))

    real_img = GESTURE_IMAGES.get(gesture)
    if real_img is not None:
        base_size = int(180 * pulse_s)
        resized = cv2.resize(real_img, (base_size, base_size))
        
        # Calculate bounding box
        x1 = cx - base_size // 2
        y1 = cy - base_size // 2
        
        # Ensure it's within bounds
        h, w = canvas.shape[:2]
        if x1 >= 0 and y1 >= 0 and x1 + base_size < w and y1 + base_size < h:
            # Mask out pure black
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            
            roi = canvas[y1:y1+base_size, x1:x1+base_size]
            roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            roi_fg = cv2.bitwise_and(resized, resized, mask=mask)
            
            # Glow circle behind
            alpha_rect(canvas, (cx-base_size//2, cy-base_size//2), (cx+base_size//2, cy+base_size//2), colour, 0.1)
            cv2.circle(canvas, (cx, cy), base_size//2, colour, int(2*s)+1, cv2.LINE_AA)
            
            canvas[y1:y1+base_size, x1:x1+base_size] = cv2.add(roi_bg, roi_fg)
            return

    # Fallback
    alpha_rect(canvas, (cx-60, cy-60), (cx+60, cy+60), colour, 0.2)
    cv2.circle(canvas, (cx, cy), 60, colour, 2)
    put_centered(canvas, gesture.upper(), cx, cy, fs=0.8, colour=colour, bold=True)


# ────────────────────────────────────────────────────────────────────────────
#  DETECTION BOX DRAWING
# ────────────────────────────────────────────────────────────────────────────

def draw_detection_box(frame, x1, y1, x2, y2, label, conf, colour):
    # Glow
    alpha_rect(frame,(x1-8,y1-8),(x2+8,y2+8), colour, 0.12)
    # Box
    round_rect(frame, (x1,y1),(x2,y2), colour, r=10, t=2)
    # Corner accents
    cl, ct = 20, 3
    for (pa, pb, pc) in [
        ((x1,y1+cl),(x1,y1),(x1+cl,y1)),
        ((x2-cl,y1),(x2,y1),(x2,y1+cl)),
        ((x1,y2-cl),(x1,y2),(x1+cl,y2)),
        ((x2-cl,y2),(x2,y2),(x2,y2-cl)),
    ]:
        cv2.line(frame,pa,pb,colour,ct); cv2.line(frame,pb,pc,colour,ct)
    # Pill label
    pill = f"  {label.upper()}  {int(conf*100)}%  "
    tw,th,bl = text_size(pill, 0.52)
    py1 = max(y1-th-16, 2); py2 = max(y1-4, th+6)
    alpha_rect(frame,(x1,py1),(x1+tw+4,py2), colour, 0.88)
    round_rect(frame,(x1,py1),(x1+tw+4,py2), colour, r=5, t=1)
    cv2.putText(frame, pill, (x1+2, py2-bl-2), FONT, 0.52, (10,10,10), 1, cv2.LINE_AA)


# ────────────────────────────────────────────────────────────────────────────
#  LAYOUT BUILDER  –  composites the full frame
# ────────────────────────────────────────────────────────────────────────────

WIN_W, WIN_H = 1280, 720
AI_PANEL_W   = 400          # left AI panel width
CAM_PANEL_W  = WIN_W - AI_PANEL_W  # right camera panel
CENTER_X     = AI_PANEL_W  # divider x


def build_frame(camera_frame, state):
    """Return a fully composited WIN_W×WIN_H frame."""
    canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    canvas[:] = C["bg_dark"]

    # ── Subtle grid texture ──────────────────────────────────────
    for gx in range(0, WIN_W, 40):
        cv2.line(canvas, (gx,0), (gx,WIN_H), C["divider"], 1)
    for gy in range(0, WIN_H, 40):
        cv2.line(canvas, (0,gy), (WIN_W,gy), C["divider"], 1)

    # ── Camera panel (right side) ────────────────────────────────
    cam_h = camera_frame.shape[0]
    cam_w = camera_frame.shape[1]
    target_h = WIN_H - 110  # leave room for top/bottom bars
    target_w = CAM_PANEL_W - 20
    scale_f = min(target_w / cam_w, target_h / cam_h)
    new_w = int(cam_w * scale_f)
    new_h = int(cam_h * scale_f)
    resized_cam = cv2.resize(camera_frame, (new_w, new_h))
    cx_off = AI_PANEL_W + (CAM_PANEL_W - new_w) // 2
    cy_off = 55 + (target_h - new_h) // 2
    canvas[cy_off:cy_off+new_h, cx_off:cx_off+new_w] = resized_cam

    # Camera panel border
    round_rect(canvas,
               (AI_PANEL_W+8, 50),
               (WIN_W-8, WIN_H-55),
               C["accent"], r=14, t=1)

    # ── AI Panel (left side) ─────────────────────────────────────
    _draw_ai_panel(canvas, state)

    # ── Top bar ──────────────────────────────────────────────────
    _draw_top_bar(canvas, state)

    # ── Bottom bar ───────────────────────────────────────────────
    _draw_bottom_bar(canvas, state)

    # ── Centre VS badge ──────────────────────────────────────────
    _draw_vs_badge(canvas, state)

    # ── Phase overlays ───────────────────────────────────────────
    ph = state["phase"]
    if ph == PHASE_RESULT:
        _draw_result_overlay(canvas, state)
    elif ph == PHASE_COUNTDOWN:
        _draw_countdown_overlay(canvas, state)
    elif ph == PHASE_GAME_OVER:
        _draw_game_over_overlay(canvas, state)

    return canvas


def _draw_top_bar(canvas, state):
    alpha_rect(canvas, (0,0), (WIN_W, 50), C["bg_mid"], 0.95)
    cv2.line(canvas, (0,50), (WIN_W,50), C["accent"], 1)

    # Title
    cv2.putText(canvas, "ROCK · PAPER · SCISSORS",
                (16, 34), FONT_BOLD, 0.75, C["accent"], 1, cv2.LINE_AA)

    # FPS
    fps_str = f"FPS {state['fps']:05.1f}   CAM {state['cam_idx']}"
    tw,_,_ = text_size(fps_str, 0.5)
    cv2.putText(canvas, fps_str, (WIN_W-tw-16, 32), FONT, 0.5, C["dim"], 1, cv2.LINE_AA)

    # Round counter (top centre)
    if state["phase"] not in (PHASE_INPUT,):
        rnd_str = f"Round  {state['round_num']} / {state['total_rounds']}"
        put_centered(canvas, rnd_str, WIN_W//2, 28, fs=0.62, colour=C["gold"])


def _draw_bottom_bar(canvas, state):
    alpha_rect(canvas, (0, WIN_H-50), (WIN_W, WIN_H), C["bg_mid"], 0.95)
    cv2.line(canvas, (0, WIN_H-50), (WIN_W, WIN_H-50), C["accent"], 1)

    # Detection readout
    dets = state.get("detections", [])
    if dets:
        x_cur = AI_PANEL_W + 20
        for (cls_name, conf) in dets[:3]:
            col = C.get(cls_name, C["dim"])
            tag = f"● {cls_name.upper()}  {int(conf*100)}%"
            cv2.putText(canvas, tag, (x_cur, WIN_H-16), FONT, 0.52, col, 1, cv2.LINE_AA)
            tw,_,_ = text_size(tag, 0.52)
            x_cur += tw + 28
    else:
        cv2.putText(canvas, "No gesture detected — show your hand",
                    (AI_PANEL_W+20, WIN_H-16), FONT, 0.5, C["dim"], 1, cv2.LINE_AA)

    # Key hints
    hints = "[SPACE] Lock    [R] Restart    [N] Camera    [ESC] Quit"
    tw,_,_ = text_size(hints, 0.42)
    cv2.putText(canvas, hints, (WIN_W-tw-16, WIN_H-16), FONT, 0.42, C["dim"], 1, cv2.LINE_AA)


def _draw_ai_panel(canvas, state):
    # Background
    alpha_rect(canvas, (0,50), (AI_PANEL_W, WIN_H-50), C["bg_panel"], 0.92)
    cv2.line(canvas, (AI_PANEL_W,50), (AI_PANEL_W, WIN_H-50), C["divider"], 2)

    # Header
    alpha_rect(canvas, (0,50), (AI_PANEL_W, 100), C["bg_mid"], 0.95)
    cv2.line(canvas, (0,100), (AI_PANEL_W,100), C["divider"], 1)
    put_centered(canvas, "A I", AI_PANEL_W//2, 75, fs=0.85, colour=C["accent"], bold=True)

    # Score box
    ps = state["player_score"]
    cs = state["computer_score"]
    dr = state["draws"]

    # AI score badge
    badge_col = C["lose"] if cs >= ps else C["dim"]
    alpha_rect(canvas, (AI_PANEL_W-80, 55), (AI_PANEL_W-8, 97), badge_col, 0.3)
    round_rect(canvas, (AI_PANEL_W-80, 55), (AI_PANEL_W-8, 97), badge_col, r=8, t=1)
    put_centered(canvas, str(cs), AI_PANEL_W-44, 76, fs=1.0, colour=C["white"], bold=True, t=2)

    # Gesture display area
    art_cx = AI_PANEL_W // 2
    art_cy = WIN_H // 2 - 20

    ph = state["phase"]
    ai_move = state.get("last_cpu_move")
    t_now   = state.get("_t", time.time())

    if ph == PHASE_PLAYING:
        # Thinking animation – cycling through gestures
        idx = int(t_now * 1.8) % 3
        ghost_move = ["rock","paper","scissors"][idx]
        col = C.get(ghost_move, C["accent"])
        # Dim ghost
        ghost_canvas = canvas.copy()
        draw_gesture_art(ghost_canvas, art_cx, art_cy, ghost_move, col, scale=1.1, pulse=t_now)
        cv2.addWeighted(ghost_canvas, 0.28, canvas, 0.72, 0, canvas)
        # "THINKING..." text
        put_centered(canvas, "THINKING...", art_cx, art_cy + 145, fs=0.65, colour=C["dim"])

    elif ph == PHASE_COUNTDOWN:
        # Still hiding
        put_centered(canvas, "CHOOSING...", art_cx, art_cy, fs=0.75, colour=C["dim"])
        # Spinning dots
        n_dots = 8
        dot_r  = 55
        for di in range(n_dots):
            angle = (t_now * 180 + di * (360/n_dots)) % 360
            rad   = math.radians(angle)
            dx = int(art_cx + dot_r * math.cos(rad))
            dy = int(art_cy + dot_r * math.sin(rad))
            alpha = (di+1) / n_dots
            col_dot = tuple(int(c * alpha) for c in C["accent"])
            cv2.circle(canvas, (dx, dy), int(5 * alpha), col_dot, -1, cv2.LINE_AA)

    elif ph in (PHASE_RESULT, PHASE_GAME_OVER) and ai_move:
        col = C.get(ai_move, C["accent"])
        # Glowing background circle
        alpha_rect(canvas, (art_cx-120, art_cy-130), (art_cx+120, art_cy+100), col, 0.08)
        cv2.circle(canvas, (art_cx, art_cy-15), 115, col, 1, cv2.LINE_AA)
        draw_gesture_art(canvas, art_cx, art_cy-15, ai_move, col, scale=1.1, pulse=t_now)
        put_centered(canvas, GESTURE_LABEL.get(ai_move,"?"), art_cx, art_cy+115,
                     fs=0.8, colour=col, bold=True)

    else:
        put_centered(canvas, "READY", art_cx, art_cy, fs=0.9, colour=C["dim"])

    # Draw mini score strip at bottom of AI panel
    strip_y = WIN_H - 50 - 70
    alpha_rect(canvas, (10, strip_y), (AI_PANEL_W-10, strip_y+60), C["bg_mid"], 0.7)
    round_rect(canvas, (10, strip_y), (AI_PANEL_W-10, strip_y+60), C["divider"], r=8, t=1)
    put_centered(canvas, f"Draws: {dr}", AI_PANEL_W//2, strip_y+30, fs=0.52, colour=C["dim"])


def _draw_vs_badge(canvas, state):
    """Animated VS badge on the divider line."""
    t_now = state.get("_t", time.time())
    pulse = 0.85 + 0.15 * abs(math.sin(t_now * 2.1))
    r = int(34 * pulse)
    bx, by = AI_PANEL_W, WIN_H // 2 - 20
    # Glow
    alpha_rect(canvas, (bx-r-10, by-r-10), (bx+r+10, by+r+10), C["accent"], 0.12)
    cv2.circle(canvas, (bx, by), r, C["bg_mid"], -1, cv2.LINE_AA)
    cv2.circle(canvas, (bx, by), r, C["accent"], 2, cv2.LINE_AA)
    put_centered(canvas, "VS", bx, by, fs=0.72, colour=C["accent"], bold=True, t=2)

    # Player score badge on divider (right side of VS)
    ps = state["player_score"]
    badge_col = C["win"] if ps >= state["computer_score"] else C["dim"]
    brs_x = bx + 45
    brs_y = by - 22
    alpha_rect(canvas, (brs_x, brs_y), (brs_x+52, brs_y+44), badge_col, 0.3)
    round_rect(canvas, (brs_x, brs_y), (brs_x+52, brs_y+44), badge_col, r=8, t=1)
    put_centered(canvas, str(ps), brs_x+26, by, fs=1.0, colour=C["white"], bold=True, t=2)
    put_centered(canvas, "YOU", brs_x+26, by-38, fs=0.45, colour=C["dim"])

    bls_x = bx - 97
    alpha_rect(canvas, (bls_x, brs_y), (bls_x+52, brs_y+44),
               C["lose"] if state["computer_score"] > ps else C["dim"], 0.3)
    round_rect(canvas, (bls_x, brs_y), (bls_x+52, brs_y+44),
               C["lose"] if state["computer_score"] > ps else C["dim"], r=8, t=1)
    put_centered(canvas, str(state["computer_score"]), bls_x+26, by,
                 fs=1.0, colour=C["white"], bold=True, t=2)
    put_centered(canvas, "AI", bls_x+26, by-38, fs=0.45, colour=C["dim"])

    # "PLAYER" label on camera side
    put_centered(canvas, "P L A Y E R",
                 AI_PANEL_W + CAM_PANEL_W//2, 76, fs=0.72, colour=C["white"], bold=True)


def _draw_countdown_overlay(canvas, state):
    t_now    = state.get("_t", time.time())
    elapsed  = t_now - state["countdown_start"]
    remain   = max(COUNTDOWN_TIME - elapsed, 0)
    lm       = state.get("locked_move","")
    col      = C.get(lm, C["accent"])

    # Countdown ring on camera side
    cx = AI_PANEL_W + CAM_PANEL_W//2
    cy = WIN_H//2
    r  = 70
    cv2.circle(canvas, (cx,cy), r+8, C["bg_dark"], -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx,cy), r+8, col, 1, cv2.LINE_AA)

    angle = int(360 * (1 - remain / COUNTDOWN_TIME))
    cv2.ellipse(canvas, (cx,cy), (r,r), -90, 0, angle, col, 6, cv2.LINE_AA)
    put_centered(canvas, f"{remain:.1f}", cx, cy, fs=1.2, colour=col, bold=True, t=2)
    put_centered(canvas, "LOCKING IN...", cx, cy+95, fs=0.62, colour=col)
    if lm:
        put_centered(canvas, lm.upper(), cx, cy+125, fs=0.55, colour=C["dim"])


def _draw_result_overlay(canvas, state):
    result   = state.get("last_result","")
    pm       = state.get("last_player_move","")
    t_now    = state.get("_t", time.time())
    elapsed  = t_now - state["result_timer"]
    remain   = max(RESULT_TIME - elapsed, 0)

    if result == "player":
        res_text = "YOU WIN!"
        res_col  = C["win"]
    elif result == "computer":
        res_text = "AI WINS!"
        res_col  = C["lose"]
    else:
        res_text = "DRAW!"
        res_col  = C["draw"]

    # Banner across camera section
    bx1 = AI_PANEL_W + 20
    bx2 = WIN_W - 20
    by1 = WIN_H//2 - 55
    by2 = WIN_H//2 + 55
    alpha_rect(canvas, (bx1,by1), (bx2,by2), C["bg_dark"], 0.82)
    round_rect(canvas, (bx1,by1), (bx2,by2), res_col, r=14, t=2)

    cxb = (bx1+bx2)//2
    put_centered(canvas, res_text, cxb, WIN_H//2-20, fs=1.3, colour=res_col, bold=True, t=2)
    sub = f"Your move: {pm.upper()}" if pm else ""
    put_centered(canvas, sub, cxb, WIN_H//2+22, fs=0.6, colour=C["dim"])

    # Countdown strip
    bar_w = bx2-bx1-40
    filled = int(bar_w * (remain / RESULT_TIME))
    alpha_rect(canvas,(bx1+20,by2+10),(bx2-20,by2+18), C["divider"], 0.8)
    if filled>0:
        alpha_rect(canvas,(bx1+20,by2+10),(bx1+20+filled,by2+18), res_col, 0.9)


def _draw_game_over_overlay(canvas, state):
    ps = state["player_score"]
    cs = state["computer_score"]
    dr = state["draws"]

    if ps > cs:
        champ = "YOU WIN THE MATCH!"
        cc    = C["win"]
    elif cs > ps:
        champ = "AI WINS THE MATCH!"
        cc    = C["lose"]
    else:
        champ = "IT'S A TIE!"
        cc    = C["draw"]

    # Full overlay dim
    alpha_rect(canvas, (0,0), (WIN_W,WIN_H), C["bg_dark"], 0.70)

    # Central panel
    pw, ph_ = 680, 340
    px = (WIN_W-pw)//2
    py = (WIN_H-ph_)//2
    alpha_rect(canvas,(px,py),(px+pw,py+ph_), C["bg_panel"], 0.97)
    round_rect(canvas,(px,py),(px+pw,py+ph_), cc, r=20, t=3)

    cxp = px+pw//2
    put_centered(canvas, "GAME  OVER", cxp, py+55, fs=1.1, colour=C["accent"], bold=True, t=2)
    put_centered(canvas, champ, cxp, py+115, fs=1.0, colour=cc, bold=True, t=2)

    score_str = f"YOU  {ps}  —  {cs}  AI      Draws: {dr}"
    put_centered(canvas, score_str, cxp, py+178, fs=0.7, colour=C["white"])

    total_str = f"Best of {state['total_rounds']} Rounds"
    put_centered(canvas, total_str, cxp, py+222, fs=0.55, colour=C["dim"])

    # Draw mini gesture summary
    if state.get("last_player_move") and state.get("last_cpu_move"):
        put_centered(canvas, "Last Round:", cxp, py+262, fs=0.5, colour=C["dim"])

    put_centered(canvas, "[R]  Play Again      [ESC]  Quit",
                 cxp, py+310, fs=0.62, colour=C["accent"])


# ────────────────────────────────────────────────────────────────────────────
#  ROUNDS INPUT  (on top of live camera feed)
# ────────────────────────────────────────────────────────────────────────────

def draw_input_screen(canvas, inp_str, inp_err, t_now):
    alpha_rect(canvas,(0,0),(WIN_W,WIN_H), C["bg_dark"], 0.72)

    pw, ph_ = 600, 300
    px = (WIN_W-pw)//2
    py = (WIN_H-ph_)//2
    alpha_rect(canvas,(px,py),(px+pw,py+ph_), C["bg_panel"], 0.97)

    pulse = 0.5 + 0.5*abs(math.sin(t_now*1.5))
    border_col = tuple(int(c*pulse + C["bg_mid"][i]*(1-pulse)) for i,c in enumerate(C["accent"]))
    round_rect(canvas,(px,py),(px+pw,py+ph_), border_col, r=18, t=2)

    cxp = px+pw//2
    put_centered(canvas,"ROCK · PAPER · SCISSORS",cxp,py+48,fs=0.9,colour=C["accent"],bold=True)
    put_centered(canvas,"VS  COMPUTER",cxp,py+85,fs=0.6,colour=C["dim"])

    put_centered(canvas,"How many rounds?",cxp,py+140,fs=0.68,colour=C["white"])

    # Input box
    ibx=px+160; iby=py+158; ibw=pw-320; ibh=50
    alpha_rect(canvas,(ibx,iby),(ibx+ibw,iby+ibh),C["bg_dark"],0.9)
    round_rect(canvas,(ibx,iby),(ibx+ibw,iby+ibh),C["accent"],r=10,t=2)
    disp = inp_str if inp_str else "  _"
    put_centered(canvas, disp, ibx+ibw//2, iby+ibh//2+2, fs=1.0,
                 colour=C["win"] if inp_str else C["dim"], bold=True, t=2)

    put_centered(canvas,"Suggested: 3 · 5 · 7 · 10",cxp,py+236,fs=0.48,colour=C["dim"])
    put_centered(canvas,"Press ENTER to start",cxp,py+268,fs=0.52,colour=C["accent"])

    if inp_err:
        put_centered(canvas, inp_err, cxp, py+ph_+28, fs=0.5, colour=C["lose"])


# ────────────────────────────────────────────────────────────────────────────
#  GAME STATE
# ────────────────────────────────────────────────────────────────────────────

PHASE_INPUT     = "input"
PHASE_PLAYING   = "playing"
PHASE_COUNTDOWN = "countdown"
PHASE_RESULT    = "result"
PHASE_GAME_OVER = "game_over"

COUNTDOWN_TIME  = 1.5    # seconds for countdown ring
RESULT_TIME     = 3.0    # seconds to show round result


def get_winner(p, c):
    if p == c: return "draw"
    return "player" if {"rock":"scissors","scissors":"paper","paper":"rock"}[p]==c else "computer"


def fresh_state(total_rounds, cam_idx):
    return {
        "phase":         PHASE_PLAYING,
        "total_rounds":  total_rounds,
        "round_num":     1,
        "player_score":  0,
        "computer_score":0,
        "draws":         0,
        "fps":           0.0,
        "cam_idx":       cam_idx,
        "detections":    [],
        "last_player_move": None,
        "last_cpu_move":    None,
        "last_result":      None,
        "result_timer":     0.0,
        "countdown_start":  0.0,
        "locked_move":      None,
        "_t":               time.time(),
    }


# ────────────────────────────────────────────────────────────────────────────
#  CAMERA
# ────────────────────────────────────────────────────────────────────────────

def get_camera(start=0):
    backend = cv2.CAP_DSHOW if hasattr(cv2,"CAP_DSHOW") else 0
    for i in range(start, 6):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            ret, fr = cap.read()
            if ret and fr is not None:
                print(f"[OK]  Camera {i}")
                return cap, i
            cap.release()
    return None, -1


# ────────────────────────────────────────────────────────────────────────────
#  LOAD MODEL
# ────────────────────────────────────────────────────────────────────────────

print("[INFO] Loading YOLO model...")
try:
    model = YOLO("model.pt")
    CLASS_NAMES = ["rock", "paper", "scissors"]
    print("[OK]  Model loaded.\n")
except Exception as e:
    print(f"[ERROR] {e}"); exit(1)

cap, cam_idx = get_camera(0)
if cap is None:
    print("[ERROR] No camera."); exit(1)

cv2.namedWindow("RPS PRO  |  Akshay Gurav", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RPS PRO  |  Akshay Gurav", WIN_W, WIN_H)

# ────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ────────────────────────────────────────────────────────────────────────────

state      = None
phase      = PHASE_INPUT
inp_str    = ""
inp_err    = ""
fps        = 0.0
prev_time  = time.time()
fail_count = 0

print("=" * 58)
print("  RPS PRO EDITION  |  by Stuti Gupta")
print("  SPACE → lock move  |  R → restart  |  N → cam  |  ESC")
print("=" * 58 + "\n")

while True:
    ret, raw_frame = cap.read()
    if not ret or raw_frame is None:
        fail_count += 1
        if fail_count > 30:
            print("[WARN] Camera stalled. Press N."); fail_count = 0
        # Show blank frame
        raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    fail_count = 0

    # FPS
    now       = time.time()
    fps       = 0.9*fps + 0.1*(1.0/max(now-prev_time, 1e-6))
    prev_time = now

    # ── YOLO  ────────────────────────────────────────────────────
    results    = model(raw_frame, stream=True, verbose=False)
    detections = []
    best_det   = None

    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2 = (int(v) for v in box.xyxy[0])
            conf  = round(float(box.conf[0]), 2)
            cls   = int(box.cls[0])
            name  = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else "unknown"
            col   = C.get(name, C["dim"])
            detections.append((name, conf))
            if best_det is None or conf > best_det[1]:
                best_det = (name, conf)
            draw_detection_box(raw_frame, x1, y1, x2, y2, name, conf, col)

    # ── INPUT PHASE (rounds selection) ───────────────────────────
    if phase == PHASE_INPUT:
        canvas = build_frame(raw_frame, {
            "phase": PHASE_INPUT, "fps": fps, "cam_idx": cam_idx,
            "round_num":1,"total_rounds":1,
            "player_score":0,"computer_score":0,"draws":0,
            "detections": detections, "_t": now,
        })
        draw_input_screen(canvas, inp_str, inp_err, now)
        cv2.imshow("RPS PRO  |  Stuti Gupta", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        elif key == 13:   # ENTER
            try:
                n = int(inp_str)
                if n < 1: raise ValueError
                state = fresh_state(n, cam_idx)
                phase = PHASE_PLAYING
                inp_str = ""; inp_err = ""
                print(f"[GAME]  {n}-round match starting!")
            except ValueError:
                inp_err = "Enter a valid number  (e.g.  3  5  10)"
        elif key == 8:  inp_str = inp_str[:-1]; inp_err = ""
        elif 48 <= key <= 57: inp_str += chr(key); inp_err = ""
        continue

    # ── Update state meta ────────────────────────────────────────
    state["fps"]        = fps
    state["cam_idx"]    = cam_idx
    state["detections"] = detections
    state["_t"]         = now

    ph = state["phase"]

    # ── PLAYING ──────────────────────────────────────────────────
    if ph == PHASE_PLAYING:
        canvas = build_frame(raw_frame, state)

        # Prompt overlay on camera side
        if best_det and best_det[0] != "unknown":
            col = C.get(best_det[0], C["accent"])
            msg = f"Detected: {best_det[0].upper()}  {int(best_det[1]*100)}%   →  SPACE to lock"
        else:
            col = C["dim"]
            msg = "Show your hand gesture to the camera..."

        cxc = AI_PANEL_W + CAM_PANEL_W//2
        cyc = WIN_H - 50 - 38
        put_centered(canvas, msg, cxc, cyc, fs=0.60, colour=col)

        cv2.imshow("RPS PRO  |  Stuti Gupta", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        elif key == ord(' '):
            if best_det and best_det[0] != "unknown":
                state["locked_move"]     = best_det[0]
                state["countdown_start"] = now
                state["phase"]           = PHASE_COUNTDOWN
            else:
                print("[WARN] No gesture detected!")
        elif key == ord('r'):
            phase = PHASE_INPUT; inp_str = ""; inp_err = ""
        elif key == ord('n'):
            cap.release()
            cap, cam_idx = get_camera(cam_idx+1)
            if cap is None: cap, cam_idx = get_camera(0)

    # ── COUNTDOWN ────────────────────────────────────────────────
    elif ph == PHASE_COUNTDOWN:
        canvas = build_frame(raw_frame, state)
        cv2.imshow("RPS PRO  |  Stuti Gupta", canvas)
        cv2.waitKey(1)

        elapsed = now - state["countdown_start"]
        if elapsed >= COUNTDOWN_TIME:
            pm  = state["locked_move"]
            cm  = random.choice(["rock","paper","scissors"])
            res = get_winner(pm, cm)

            state["last_player_move"] = pm
            state["last_cpu_move"]    = cm
            state["last_result"]      = res
            state["result_timer"]     = now
            state["phase"]            = PHASE_RESULT

            if res == "player":   state["player_score"]   += 1
            elif res == "computer": state["computer_score"] += 1
            else:                 state["draws"]           += 1

            print(f"[R{state['round_num']}]  YOU: {pm:8s}  AI: {cm:8s}  → {res.upper()}")

    # ── RESULT ───────────────────────────────────────────────────
    elif ph == PHASE_RESULT:
        canvas = build_frame(raw_frame, state)
        cv2.imshow("RPS PRO  |  Stuti Gupta", canvas)
        cv2.waitKey(1)

        if now - state["result_timer"] >= RESULT_TIME:
            state["round_num"] += 1
            if state["round_num"] > state["total_rounds"]:
                state["phase"] = PHASE_GAME_OVER
                ps = state["player_score"]; cs2 = state["computer_score"]
                verdict = "YOU WIN" if ps>cs2 else "AI WINS" if cs2>ps else "DRAW"
                print(f"\n[MATCH OVER]  {verdict}  —  YOU {ps}  AI {cs2}  Draws {state['draws']}\n")
            else:
                state["phase"] = PHASE_PLAYING

    # ── GAME OVER ────────────────────────────────────────────────
    elif ph == PHASE_GAME_OVER:
        canvas = build_frame(raw_frame, state)
        cv2.imshow("RPS PRO  |  Stuti Gupta", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        elif key == ord('r'):
            phase = PHASE_INPUT; inp_str = ""; inp_err = ""
        elif key == ord('n'):
            cap.release()
            cap, cam_idx = get_camera(cam_idx+1)
            if cap is None: cap, cam_idx = get_camera(0)


cap.release()
cv2.destroyAllWindows()
print("\n[DONE]  Thanks for playing!  —  Stuti Gupta\n")
