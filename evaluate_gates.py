#!/usr/bin/env python3
"""
Batch metrics on image folders: pole states, PnP, reprojection error.

  python3 evaluate_gates.py --folder images/image_navigation_01
  python3 evaluate_gates.py --folder images/image_qualification_01 --qualification
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

from gate_detection_core import detect_gate_with_state
from gate_pose_pnp import camera_matrix_from_fov, estimate_gate_pose


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--folder", required=True)
    p.add_argument("--qualification", action="store_true")
    p.add_argument("--fov", type=float, default=60.0)
    p.add_argument("--gate-width", type=float, default=1.5)
    args = p.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    folder = args.folder
    if not os.path.isabs(folder):
        folder = os.path.join(root, folder)
    if not os.path.isdir(folder):
        print("Not found:", folder, file=sys.stderr)
        sys.exit(1)

    names = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
    n_two = n_one = n_none = 0
    reprojs: list[float] = []
    yaw_ok: list[float] = []

    for name in names:
        path = os.path.join(folder, name)
        im = cv2.imread(path)
        if im is None:
            continue
        h, w = im.shape[:2]
        st = detect_gate_with_state(im, use_color_boost=args.qualification)
        if st.state == "two":
            n_two += 1
            if st.filtered_edges is not None:
                K = camera_matrix_from_fov(w, h, args.fov)
                pose = estimate_gate_pose(
                    im,
                    st.left_x,
                    st.right_x,
                    st.filtered_edges,
                    st.roi_y0,
                    K,
                    width_m=args.gate_width,
                    height_m=None,
                )
                if pose.ok:
                    reprojs.append(pose.reproj_err_px)
                    if pose.reproj_err_px <= 25.0:
                        yaw_ok.append(abs(pose.yaw_error_deg))
        elif st.state == "one":
            n_one += 1
        else:
            n_none += 1

    n = n_two + n_one + n_none
    print(f"Folder: {folder}")
    print(f"Images: {n}  |  two={n_two}  one={n_one}  none={n_none}")
    if reprojs:
        print(
            f"PnP reproj (px): mean={np.mean(reprojs):.2f}  "
            f"med={np.median(reprojs):.2f}  max={np.max(reprojs):.2f}"
        )
        print(
            f"Frames with reproj<=25: {len(yaw_ok)} / {n_two} two-pole frames"
        )
        if yaw_ok:
            print(f"|yaw_err| deg (when reproj ok): mean={np.mean(yaw_ok):.2f}")

    print("\nFurther improvement ideas:")
    print("- Tune --fov to your camera; wrong FOV blows PnP even with aspect fix.")
    print("- Calibrate K,D in-water; replace camera_matrix_from_fov.")
    print("- If 'one' is high: adjust min_sep_frac/max_sep_frac in gate_detection_core.")
    print("- For live: use live_gate_detector.py --temporal 0.35")


if __name__ == "__main__":
    main()
