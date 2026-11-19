"""
Interactive HSV threshold tuner.

Run:  python hsv_tuner.py [camera_index | image_path]

Drag the sliders until ONLY the tape you care about shows white in the mask.
Tune under the same lighting the robot will run in.

Keys:  m        switch between tuning YELLOW and BLUE
       s        save both ranges to hsv_config.json (used by track_test.py)
       q / Esc  quit (without saving)

Tip: hold a piece of tape up to the camera, or point it at the track on the floor.
     You can also pass a snapshot saved from track_test.py instead of a camera index.
"""

import os
import sys

import cv2
import numpy as np

from perception import load_config, save_config, open_camera

WIN = "HSV tuner  (m=switch color, s=save, q=quit)"
SLIDERS = [("H min", 179), ("S min", 255), ("V min", 255),
           ("H max", 179), ("S max", 255), ("V max", 255)]


def set_sliders(lo, hi):
    for (name, _), val in zip(SLIDERS, list(lo) + list(hi)):
        cv2.setTrackbarPos(name, WIN, int(val))


def get_sliders():
    vals = [cv2.getTrackbarPos(name, WIN) for name, _ in SLIDERS]
    return vals[:3], vals[3:]


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    still = None
    cap = None
    if os.path.isfile(arg):
        still = cv2.imread(arg)
        if still is None:
            sys.exit(f"Could not read image {arg}")
    else:
        cap = open_camera(int(arg))

    cfg = load_config()
    color = "yellow"

    cv2.namedWindow(WIN)
    for name, maxval in SLIDERS:
        cv2.createTrackbar(name, WIN, 0, maxval, lambda _: None)
    set_sliders(cfg[f"{color}_lo"], cfg[f"{color}_hi"])
    print(__doc__)

    while True:
        if still is not None:
            frame = still.copy()
        else:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera")
                break
        frame = cv2.resize(frame, (480, 360))

        lo, hi = get_sliders()
        cfg[f"{color}_lo"], cfg[f"{color}_hi"] = lo, hi

        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        masked = cv2.bitwise_and(frame, frame, mask=mask)

        label = f"Tuning {color.upper()}  lo={lo} hi={hi}"
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        top = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.imshow(WIN, np.vstack([top, np.hstack([masked, np.zeros_like(masked)])]))

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("m"):
            color = "blue" if color == "yellow" else "yellow"
            set_sliders(cfg[f"{color}_lo"], cfg[f"{color}_hi"])
            print(f"Now tuning {color}")
        elif key == ord("s"):
            save_config(cfg)
            print(f"Saved to hsv_config.json: {cfg}")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
