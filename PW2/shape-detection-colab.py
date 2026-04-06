import cv2
import numpy as np
import json
import os
from collections import defaultdict

# ===== CONFIG KNOBS (tweak me daddy) =====
SHAPE_MIN_AREA     = 1500    # no tiny lil babies
SHAPE_POLY_EPSILON = 0.02    # contour approximation wiggle room

# the rainbow dictionary (HSV ranges) — tightened to actual sampled values
SHAPE_COLOR_RANGES = {
    "Red": [
        (np.array([0,   120, 50]), np.array([8,   255, 255])),  # H=0-2, S drops to 140 on highlights
        (np.array([172, 120, 50]), np.array([179, 255, 255]))   # upper red, no purple stealing
    ],
    "Orange": [
        (np.array([10, 150, 50]), np.array([18, 255, 255]))     # H=12, S=191-201 sampled
    ],
    "Yellow": [
        (np.array([22, 150, 50]), np.array([35, 255, 255]))     # H=28, S=200-222 sampled
    ],
    "Green": [
        (np.array([35, 40, 40]), np.array([87, 255, 255]))
    ],
    "Teal": [
        (np.array([88, 130, 50]), np.array([100, 255, 255]))    # H=94, S=170-192 sampled
    ],
    "Blue": [
        (np.array([100, 120, 50]), np.array([125, 255, 255]))   # H=109, S=144-186 sampled
    ],
    "Purple": [
        (np.array([150, 100, 80]), np.array([175, 255, 255]))   # H=160-167, S=128-138 sampled
    ],
}

# ===== LOAD CALIBRATION (overrides hardcoded ranges if file exists) =====
CALIBRATION_PATH = "./colour_ranges.json"

def _load_calibration():
    """slurp up the saved calibration json if it exists."""
    global SHAPE_COLOR_RANGES
    if not os.path.exists(CALIBRATION_PATH):
        print("[shape_detector] no calibration file found, using hardcoded ranges.")
        return
    try:
        with open(CALIBRATION_PATH, "r") as f:
            raw = json.load(f)
        # convert lists back to numpy arrays
        loaded = {}
        for colour, ranges in raw.items():
            loaded[colour] = [
                (np.array(lo), np.array(hi)) for lo, hi in ranges
            ]
        SHAPE_COLOR_RANGES.update(loaded)
        print(f"[shape_detector] loaded calibration for: {list(loaded.keys())}")
    except Exception as e:
        print(f"[shape_detector] calibration load failed ({e}), using hardcoded ranges.")

_load_calibration()  # runs on import, no extra setup needed (others become "Unknown")
VALID_DETECTIONS = {
    ("Yellow",  "Star"),
    ("Purple",  "Diamond"),
    ("Orange",  "Cross"),
    ("Purple",  "Trapezoid"),
    ("Blue",    "Pac-Man"),
    ("Red",     "Semi-circle"),
    ("Teal",    "Octagon"),
}

# ===== STABILITY TRACKER =====
# shapes must appear in N consecutive frames before being confirmed
# and must disappear for N frames before being cleared
CONFIRM_FRAMES = 5   # frames needed to confirm a detection
CLEAR_FRAMES   = 5   # frames of absence before clearing

class StabilityTracker:
    """
    stops shapes from flashing on and off like a broken neon sign.
    only reports a shape once it's been consistently seen for CONFIRM_FRAMES.
    """
    def __init__(self):
        self._seen_count    = defaultdict(int)   # how many consecutive frames seen
        self._absent_count  = defaultdict(int)   # how many consecutive frames absent
        self._confirmed     = set()              # currently confirmed detections
        self._last_reported = {}                 # key -> last detection dict

    def update(self, raw_detections):
        """
        feed in raw detections from detect_shapes (before stability filter).
        returns only confirmed, stable detections.
        """
        current_keys = set()

        for det in raw_detections:
            key = (det["colour"], det["shape"])
            current_keys.add(key)
            self._seen_count[key]   += 1
            self._absent_count[key]  = 0
            self._last_reported[key] = det

            # promote to confirmed once seen enough times in a row
            if self._seen_count[key] >= CONFIRM_FRAMES:
                self._confirmed.add(key)

        # handle shapes that weren't seen this frame
        for key in list(self._seen_count.keys()):
            if key not in current_keys:
                self._seen_count[key]   = 0
                self._absent_count[key] += 1
                # demote from confirmed after too many absent frames
                if self._absent_count[key] >= CLEAR_FRAMES:
                    self._confirmed.discard(key)

        # return confirmed detections only
        return [self._last_reported[k] for k in self._confirmed if k in self._last_reported]

    def reset(self):
        """nuke everything — call when starting a new detection session."""
        self.__init__()


# module-level tracker instance (one per import, persists across frames)
_tracker = StabilityTracker()


def get_tracker():
    """get the module-level tracker if you need to reset it externally."""
    return _tracker


# ===== WHO'S THAT SHAPE? =====

def classify_shape(cnt):
    """geometrically interrogate a contour until it confesses its shape."""
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return "Unknown"

    circularity = 4 * np.pi * area / (peri * peri)

    hull         = cv2.convexHull(cnt)
    hull_area    = cv2.contourArea(hull)
    solidity     = area / hull_area if hull_area > 0 else 0

    x, y, w, h  = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 1
    extent       = area / (w * h) if (w * h) > 0 else 0

    approx   = cv2.approxPolyDP(cnt, SHAPE_POLY_EPSILON * peri, True)
    vertices = len(approx)

    # count convexity defects (the dents)
    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    num_defects, deep_defects = 0, 0
    try:
        defects = cv2.convexityDefects(cnt, hull_idx)
        if defects is not None:
            for d in defects[:, 0]:
                depth = d[3] / 256.0
                if depth > 3:  num_defects  += 1
                if depth > 8:  deep_defects += 1
    except cv2.error:
        pass

    # === WHO IS IT (weirdest first) ===

    # spiky or denty shapes
    if deep_defects >= 3:
        if solidity < 0.65:
            return "Star"       # pointy spike boi
        else:
            return "Cross"      # chonky plus

    # pac-man: one big bite taken out
    if deep_defects == 1 and num_defects >= 1 and 0.65 < solidity < 0.92:
        return "Pac-Man"        # waka waka

    # fairly convex stuff from here down
    if solidity > 0.88:

        # round things
        if circularity > 0.82:
            if vertices >= 7:
                return "Octagon"
            return "Circle"

        # 4-sided identity crisis — check BEFORE semicircle so diamonds don't get stolen
        if 4 <= vertices <= 6:
            # use actual vertex positions to detect trapezoid vs diamond
            # trapezoid: top edge much shorter than bottom edge (asymmetric)
            # diamond: all edges roughly equal (symmetric rotated square)
            if vertices == 4:
                pts = approx.reshape(4, 2).astype(float)
                # sort vertices by y (top to bottom)
                pts = pts[np.argsort(pts[:, 1])]
                top_pts    = pts[:2]   # two highest points
                bottom_pts = pts[2:]   # two lowest points
                top_width    = abs(top_pts[0][0]    - top_pts[1][0])
                bottom_width = abs(bottom_pts[0][0] - bottom_pts[1][0])
                # trapezoid has noticeably different top vs bottom width
                width_ratio = min(top_width, bottom_width) / max(top_width, bottom_width) if max(top_width, bottom_width) > 0 else 1
                if width_ratio < 0.75:   # top and bottom differ by >25% → trapezoid
                    return "Trapezoid"

            # diamond: AR > 1.3 (rotated square has equal diagonals so bbox tends wider)
            # or low extent (pointy corners)
            if aspect_ratio > 1.3 or extent < 0.72:
                return "Diamond"
            elif extent < 0.85:
                return "Trapezoid"
            else:
                return "Square"

        # major segment: kinda round but has a flat edge
        # only fires if it wasn't caught as a 4-sided shape above
        # aspect_ratio check removed — shape lies on floor so orientation varies
        if 0.50 <= circularity <= 0.82 and vertices >= 4:
            return "Semi-circle"

    # triangle: pointy, few vertices, not very full
    if 3 <= vertices <= 4 and solidity > 0.80 and circularity < 0.65:
        return "Triangle"

    return "Unknown"            # gave up, tbh


# ===== THE MAIN DETECTIVE FUNCTION =====

def detect_shapes(frame):
    """
    Takes a BGR frame, returns (annotated_frame, list_of_detections).

    Each detection dict has: colour, shape, bbox (x,y,w,h), area.
    Only detections in VALID_DETECTIONS are kept (or all if you remove the filter).
    """
    # frame is RGB (picamera2 outputs RGB despite BGR888 label, classic)
    hsv    = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    detected = []

    for colour_name, ranges in SHAPE_COLOR_RANGES.items():
        # build combined mask for this colour
        mask = None
        for lo, hi in ranges:
            m    = cv2.inRange(hsv, lo, hi)
            mask = m if mask is None else cv2.bitwise_or(mask, m)

        # clean up noise & fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < SHAPE_MIN_AREA:
                continue

            shape = classify_shape(cnt)

            x, y, w, h = cv2.boundingRect(cnt)

            # optional: only log things we actually care about
            # remove this check if u want literally everything
            # DEBUG: print purple regardless so we can see what shape it thinks it is
            if colour_name == "Purple":
                hull_a = cv2.contourArea(cv2.convexHull(cnt))
                sol = area / hull_a if hull_a > 0 else 0
                peri_d = cv2.arcLength(cnt, True)
                circ = 4 * np.pi * area / (peri_d * peri_d) if peri_d > 0 else 0
                approx_d = cv2.approxPolyDP(cnt, 0.02 * peri_d, True)
                print(f"  [DEBUG PURPLE] classified as '{shape}' "
                      f"area={area:.0f} bbox=({x},{y},{w},{h}) "
                      f"AR={w/h:.2f} solidity={sol:.2f} circ={circ:.2f} verts={len(approx_d)}")
            if (colour_name, shape) not in VALID_DETECTIONS:
                continue

            detected.append({
                "colour": colour_name,
                "shape":  shape,
                "bbox":   (x, y, w, h),
                "area":   area,
            })

    # run raw detections through stability filter
    stable = _tracker.update(detected)

    # only draw confirmed stable detections
    for det in stable:
        x, y, w, h = det["bbox"]
        label = f"{det['colour']} {det['shape']}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 128, 255), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 64, 255), 2)

    return frame, stable


# ===== STANDALONE TEST MODE (run this file directly to test) =====

if __name__ == "__main__":
    from picamera2 import Picamera2
    import time

    print("Shape detector standalone test. Press 'q' to quit.")

    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "BGR888"}
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(1)
    try:
        while True:
            frame = picam2.capture_array()

            vis, shapes = detect_shapes(frame.copy())

            for s in shapes:
                # print(f"  [SHAPE] {s['colour']} {s['shape']} (area={s['area']:.0f})")
                print(f" Currently Detecting a {s['colour']} {s['shape']}")
            cv2.imshow("Preview", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected, stopping program...")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
