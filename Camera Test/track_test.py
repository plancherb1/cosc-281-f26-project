"""
Live webcam test of the track perception.

Run:  python track_test.py [camera_index]

Left panel:  camera view with ROI box, image center line, track centroid,
             error bar and blue warnings.
Right panel: what the robot "sees" (yellow mask in yellow, blue mask in blue).

Keys:  q / Esc  quit
       s        save a snapshot (useful for tuning offline)
       r        reload hsv_config.json (after running hsv_tuner.py)
"""

import sys
import time

import cv2
import numpy as np

from perception import Perception, load_config, open_camera

# Simple P-controller preview, just to show what the motors would be told.
BASE_SPEED = 0.5
K_P = 0.5
SCALE = 2  # upscale the 320x240 processed image for display


def draw_overlay(r, left_cmd, right_cmd, fps):
    img = r.frame.copy()
    h, w = img.shape[:2]

    # ROI box and image center line
    cv2.rectangle(img, (0, r.roi_top), (w - 1, h - 1), (0, 255, 0), 1)
    cv2.line(img, (w // 2, r.roi_top), (w // 2, h), (255, 255, 255), 1)

    # Centroid
    if r.found:
        cy = (r.roi_top + h) // 2
        cv2.circle(img, (int(r.cx), cy), 6, (0, 0, 255), -1)
        cv2.line(img, (w // 2, cy), (int(r.cx), cy), (0, 0, 255), 2)

    img = cv2.resize(img, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)
    W = w * SCALE

    # Error bar across the top
    bar_y, bar_h = 10, 14
    cv2.rectangle(img, (10, bar_y), (W - 10, bar_y + bar_h), (60, 60, 60), -1)
    mid = W // 2
    x = int(mid + r.error * (W / 2 - 10))
    color = (0, 200, 0) if r.found else (0, 140, 255)
    cv2.rectangle(img, (min(mid, x), bar_y), (max(mid, x), bar_y + bar_h), color, -1)
    cv2.line(img, (mid, bar_y - 3), (mid, bar_y + bar_h + 3), (255, 255, 255), 1)

    status = "TRACK" if r.found else "LOST (using last error)"
    lines = [
        f"{status}   error={r.error:+.2f}   yellow px={r.yellow_pixels}",
        f"blue L={r.blue_left:.1%}  R={r.blue_right:.1%}",
        f"motors L={left_cmd:+.2f}  R={right_cmd:+.2f}   {fps:.0f} fps",
    ]
    for i, text in enumerate(lines):
        y = 50 + i * 24
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if r.blue_warning:
        side = "LEFT" if r.blue_left > r.blue_right else "RIGHT"
        text = f"BLUE WARNING ({side})"
        cv2.putText(img, text, (10, h * SCALE - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(img, text, (10, h * SCALE - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 80, 0), 2)
    return img


def draw_masks(r):
    h, w = r.yellow_mask.shape
    vis = np.zeros((h, w, 3), np.uint8)
    vis[r.yellow_mask > 0] = (0, 255, 255)
    vis[r.blue_mask > 0] = (255, 80, 0)
    # Dim everything above the ROI, since it's ignored for measurements
    vis[: r.roi_top] //= 3
    cv2.line(vis, (0, r.roi_top), (w, r.roi_top), (0, 255, 0), 1)
    return cv2.resize(vis, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)


def main():
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = open_camera(cam_index)
    perception = Perception()
    print(__doc__)

    fps, last_t, last_print = 0.0, time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera")
            break

        r = perception.process(frame)

        # What a P-controller would send to the treads (clamped to [-1, 1])
        left_cmd = float(np.clip(BASE_SPEED + K_P * r.error, -1, 1))
        right_cmd = float(np.clip(BASE_SPEED - K_P * r.error, -1, 1))

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_t, 1e-6))
        last_t = now

        if now - last_print > 0.5:
            print(f"found={r.found!s:5}  error={r.error:+.2f}  "
                  f"blueL={r.blue_left:.2f} blueR={r.blue_right:.2f}  warn={r.blue_warning}")
            last_print = now

        view = np.hstack([draw_overlay(r, left_cmd, right_cmd, fps), draw_masks(r)])
        cv2.imshow("Track test  (q=quit, s=snapshot, r=reload config)", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            name = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(name, frame)
            print(f"Saved {name}")
        elif key == ord("r"):
            perception.set_config(load_config())
            print("Reloaded hsv_config.json")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
