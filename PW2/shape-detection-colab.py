from picamera2 import Picamera2
import numpy as np
import libcamera
import cv2
import time
import math


def classify_shape(cnt):
    """is it a star? is it a trapezoid? let's find out."""
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return "Unknown"

    # math time
    circularity = 4 * np.pi * area / (peri * peri)
    epsilon = 0.03

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    x, y, w, h = cv2.boundingRect(cnt)
    extent = area / (w * h) if (w * h) > 0 else 0

    approx = cv2.approxPolyDP(cnt, epsilon * peri, True)
    vertices = len(approx)

    # dented
    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    num_defects, deep_defects = 0, 0
    try:
        defects = cv2.convexityDefects(cnt, hull_idx)
        if defects is not None:
            for d in defects[:, 0]:
                depth = d[3] / 256.0
                if depth > 3:
                    num_defects += 1
                if depth > 8:
                    deep_defects += 1
    except cv2.error:
        pass

    # === WHO'S THAT POKEMON? (weirdest shapes first) ===

    # stars & crosses dentmaxxing
    if deep_defects >= 3:
        if solidity < 0.65:
            return "Star"  # spiky boi
        else:
            return "Cross"  # chonky plus sign

    # pac-man big mouth
    if deep_defects == 1 and 0.65 < solidity < 0.92:
        return "Pac-Man"  # waka waka

    # pointy boi with few vertices
    if 3 <= vertices <= 4 and solidity > 0.85 and circularity < 0.7:
        if extent < 0.65:
            return "Triangle"  # dorito

    # the boring convex shapes
    if solidity > 0.90:

        # round-ish things
        if circularity > 0.82:
            if vertices >= 7:
                return "Octagon"
            return "Circle"

        # flat edge ruins your circularity score
        if 0.60 <= circularity <= 0.80 and vertices > 5:
            return "Semicircle"  # kinda circle

        # the 4-sided identity crisis
        if 4 <= vertices <= 6:
            if extent < 0.65:
                return "Diamond"
            elif 0.65 <= extent < 0.85:
                return "Trapezoid"
            elif extent >= 0.85:
                return "Square"  # we dont even have this btw, reg shapes jic

    return "Unknown"  # ten uning


def detect_shapes(frame):
    """colour mask everything, find contours, interrogate them."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    detected = []
    mask = None

    for colour_name, ranges in SHAPE_COLOR_RANGES.items():
        mask = None
        for lo, hi in ranges:
            m = cv2.inRange(hsv, lo, hi)
            mask = m if mask is None else cv2.bitwise_or(mask, m)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # type: ignore
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < SHAPE_MIN_AREA:
                continue

            shape = classify_shape(cnt)

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{shape}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 2)

            detected.append({
                "colour": colour_name,
                "shape": shape,
                "bbox": (x, y, w, h),
                "area": area
            })

    return frame, detected, mask


cam_size_x = 640
cam_size_y = 480

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={
    "format": "RGB888",
    "size": (cam_size_x, cam_size_y)
    },
    transform=libcamera.Transform(hflip=1, vflip=1))  # type: ignore
picam2.configure(config)
picam2.start()
time.sleep(2)

print("Processing live feed. Press 'q' in the terminal window to quit.")

# %age of the top half of the frame to discard to get the ROI
frame_discard_percentage = 0.4
# %age by which the ROI is being moved upwards
frame_discard_offset = 0.125

threshval = -1

SHAPE_MIN_AREA = 1500
SHAPE_POLY_EPSILON = 0.02

SHAPE_COLOR_RANGES = {
    "Red": [(np.array([0, 100, 100]), np.array([10, 255, 255])),
    (np.array([160, 100, 100]), np.array([179, 255, 255]))],
    "Orange": [(np.array([10, 100, 100]), np.array([20, 255, 255]))],
    "Yellow": [(np.array([25, 100, 100]), np.array([35, 255, 255]))],
    "Green": [(np.array([35, 50, 50]), np.array([85, 255, 255]))],
    "Teal": [(np.array([85, 45, 80]), np.array([100, 255, 255]))],
    "Blue": [(np.array([100, 100, 50]), np.array([130, 255, 255]))],
    "Purple": [(np.array([130, 50, 50]), np.array([160, 255, 255]))],
}

try:
    while True:
        # Capture a still frame from the camera
        frame = picam2.capture_array()
        vis = frame.copy()

        vis, shapes, mask = detect_shapes(vis)
        if shapes:
            for s in shapes:
                print(f"  [SHAPE] {s['colour']} {s['shape']} "
                    f"(area={s['area']})")

        cv2.imshow('Mask', mask)
        cv2.imshow('Shape Detect', vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nKeyboard interrupt detected, stopping program...")
finally:
    picam2.stop()
    cv2.destroyAllWindows()
