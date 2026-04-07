"""
calibratecolours.py  —  click on shapes, save HSV ranges, profit

HOW TO USE:
  1. Run this script: python3 calibrate_colours.py
  2. A menu appears — pick a colour to calibrate (e.g. press '1' for Red)
  3. Click directly on that coloured shape in the live camera window
  4. Click a few different spots on the same shape (different lighting areas)
  5. The range auto-expands to cover all your clicks + some padding
  6. Press 's' to save and move to the next colour
  7. Press 'r' to reset samples for the current colour and try again
  8. Press 'q' when done — saves colour_ranges.json to this folder
  9. shape_detector.py will automatically load the saved file next run
"""

import cv2
import numpy as np
import json
import os
import time
from picamera2 import Picamera2

# ===== CONFIG =====
SAVE_PATH   = "./colour_ranges_line.json" #has to be in same spot as shape detection code
HSV_PADDING = 15    # wiggle room added around sampled values (hue)
SAT_PADDING = 40    # wiggle room for saturation
VAL_PADDING = 50    # wiggle room for value/brightness
SAT_MIN     = 30    # never go below this saturation (avoids catching grey/white)
VAL_MIN     = 30    # never go below this value

# colours to calibrate in order
# Red needs two ranges (wraps around 0/179 in HSV), everyone else gets one
COLOURS_TO_CALIBRATE = [
    "Red",
    "Yellow",
]

# ===== GLOBALS =====
current_samples = []   # list of HSV pixels clicked so far
frame_display   = None # latest frame for drawing on


# ===== MOUSE CALLBACK =====

def on_mouse_click(event, x, y, flags, param):
    """when u click, sample that pixel's HSV. ez."""
    global current_samples, frame_display

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # sample a small patch around the click (5x5) for robustness
    hsv_frame = param["hsv"]
    h, w      = hsv_frame.shape[:2]

    x1, y1 = max(0, x - 2), max(0, y - 2)
    x2, y2 = min(w, x + 3), min(h, y + 3)
    patch   = hsv_frame[y1:y2, x1:x2].reshape(-1, 3)

    for px in patch:
        current_samples.append(px.tolist())

    print(f"  sampled pixel at ({x},{y}) → HSV {hsv_frame[y, x].tolist()} "
          f"  (total samples: {len(current_samples)})")


# ===== RANGE BUILDER =====

def samples_to_range(samples, colour_name):
    """
    Turn a pile of HSV sample points into (lo, hi) range(s).
    Red is special — it wraps around the hue wheel so gets two ranges.
    """
    if not samples:
        return None

    arr = np.array(samples)
    h_vals = arr[:, 0]
    s_vals = arr[:, 1]
    v_vals = arr[:, 2]

    s_lo = max(SAT_MIN, int(s_vals.min()) - SAT_PADDING)
    s_hi = min(255,     int(s_vals.max()) + SAT_PADDING)
    v_lo = max(VAL_MIN, int(v_vals.min()) - VAL_PADDING)
    v_hi = min(255,     int(v_vals.max()) + VAL_PADDING)

    if colour_name == "Red":
        # red wraps: hues near 0 AND near 179 are both "red"
        # split into two ranges at the wrap point
        lo1 = np.array([0,            s_lo, v_lo])
        hi1 = np.array([min(10, int(h_vals.max()) + HSV_PADDING), s_hi, v_hi])
        lo2 = np.array([max(168, int(h_vals.min()) - HSV_PADDING), s_lo, v_lo])
        hi2 = np.array([179, s_hi, v_hi])
        return [(lo1.tolist(), hi1.tolist()), (lo2.tolist(), hi2.tolist())]
    else:
        h_lo = max(0,   int(h_vals.min()) - HSV_PADDING)
        h_hi = min(179, int(h_vals.max()) + HSV_PADDING)
        lo   = np.array([h_lo, s_lo, v_lo])
        hi   = np.array([h_hi, s_hi, v_hi])
        return [(lo.tolist(), hi.tolist())]


# ===== DRAW OVERLAY =====

def draw_overlay(frame, colour_name, samples, current_range):
    """draw helpful info on the frame so u know what's happening."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # show live mask if we have samples
    if current_range:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = None
        for lo, hi in current_range:
            m    = cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        # tint matched areas cyan-ish
        tint         = np.zeros_like(vis)
        tint[:, :, 1] = 200
        vis[mask > 0] = cv2.addWeighted(vis, 0.4, tint, 0.6, 0)[mask > 0]

    # instructions panel
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

    cv2.putText(vis, f"Calibrating: {colour_name}  ({len(samples)} samples)",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(vis, "CLICK on the shape  |  S = save  |  R = reset  |  Q = quit",
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(vis, "Cyan tint = currently matched pixels",
                (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 255, 150), 1)

    return vis


# ===== MAIN =====

def main():
    global current_samples

    # --- camera init ---
    picam2 = Picamera2()
    cfg    = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "BGR888"}
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(1)
    print("Camera ready. let's calibrate!\n")

    # --- load existing calibration if present (so we can update just one colour) ---
    saved_ranges = {}
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "r") as f:
            saved_ranges = json.load(f)
        print(f"Loaded existing calibration from {SAVE_PATH}")
        print(f"  existing colours: {list(saved_ranges.keys())}\n")

    cv2.namedWindow("Calibration")

    # --- loop through each colour ---
    for colour_name in COLOURS_TO_CALIBRATE:
        current_samples = []
        print(f"\n{'='*45}")
        print(f"  NOW CALIBRATING: {colour_name}")
        print(f"  Click on the {colour_name} shape in the window.")
        print(f"  S = save this colour | R = reset | Q = quit early")
        print(f"{'='*45}")

        # set up mouse callback with access to latest HSV frame
        mouse_data = {"hsv": None}
        cv2.setMouseCallback("Calibration", on_mouse_click, mouse_data)

        saved_this_colour = False

        while not saved_this_colour:
            frame = picam2.capture_array()

            # compute HSV and give to mouse callback
            hsv              = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            mouse_data["hsv"] = hsv

            # build current range from samples so far
            current_range = samples_to_range(current_samples, colour_name)

            # draw overlay and show
            vis = draw_overlay(frame, colour_name, current_samples, current_range)
            cv2.imshow("Calibration", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') or key == ord('S'):
                if not current_samples:
                    print("  no samples yet! click on the shape first.")
                    continue
                # save this colour's range
                saved_ranges[colour_name] = [
                    (lo, hi) for lo, hi in current_range
                ]
                print(f"  saved {colour_name}: {current_range}")
                saved_this_colour = True

            elif key == ord('r') or key == ord('R'):
                current_samples = []
                print(f"  reset! start clicking again for {colour_name}.")

            elif key == ord('q') or key == ord('Q'):
                print("\n  quit early — saving what we have so far.")
                if current_samples:
                    current_range = samples_to_range(current_samples, colour_name)
                    saved_ranges[colour_name] = [(lo, hi) for lo, hi in current_range]
                _save_and_exit(saved_ranges, picam2)
                return

        print(f"  {colour_name} done! moving to next colour...")
        time.sleep(0.3)

    # --- all done! ---
    _save_and_exit(saved_ranges, picam2)


def _save_and_exit(saved_ranges, picam2):
    """dump ranges to json and say bye."""
    with open(SAVE_PATH, "w") as f:
        json.dump(saved_ranges, f, indent=2)
    print(f"\nCalibration saved to: {SAVE_PATH}")
    print("shape_detector.py will load this automatically next run.")
    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
