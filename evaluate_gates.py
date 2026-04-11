#!/usr/bin/env python3
"""
Batch metrics on image folders: pole states, PnP, reprojection error.

  python3 evaluate_gates.py --folder images/image_navigation_01
  python3 evaluate_gates.py --folder images/image_qualification_01 --qualification
  python3 evaluate_gates.py --folder images/image_navigation_01 --viz --delay 300
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

from gate_draw import render_column_strength_bar
from gate_pipeline import PipelineConfig, process_frame


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate gate detection on a folder of PNGs")
    p.add_argument("--folder", required=True)
    p.add_argument("--qualification", action="store_true")
    p.add_argument("--fov", type=float, default=60.0)
    p.add_argument("--gate-width", type=float, default=1.5)
    p.add_argument(
        "--edge-ignore",
        type=float,
        default=0.04,
        dest="edge_ignore",
        help="Column histogram border ignore fraction (same as live detector).",
    )
    p.add_argument(
        "--viz",
        action="store_true",
        help="Show detection overlay, edges, and column-strength debug (same pipeline as live).",
    )
    p.add_argument(
        "--delay",
        type=int,
        default=400,
        help="Milliseconds between frames when --viz is set.",
    )
    p.add_argument("--no-pnp", action="store_true", help="Skip PnP in viz mode (faster).")
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

    cfg = PipelineConfig(
        use_color_boost=args.qualification,
        horizontal_fov_deg=args.fov,
        gate_width_m=args.gate_width,
        use_pnp=not args.no_pnp,
        edge_ignore_frac=args.edge_ignore,
    )

    if args.viz:
        cv2.namedWindow("evaluate: Gate", cv2.WINDOW_NORMAL)
        cv2.namedWindow("evaluate: Edges", cv2.WINDOW_NORMAL)
        cv2.namedWindow("evaluate: Columns", cv2.WINDOW_NORMAL)

    for name in names:
        path = os.path.join(folder, name)
        im = cv2.imread(path)
        if im is None:
            continue
        w = im.shape[1]

        st, pose, disp = process_frame(im, cfg, None)

        if args.viz:
            if st.filtered_edges is not None:
                cv2.imshow(
                    "evaluate: Edges",
                    cv2.resize(st.filtered_edges, (640, 360)),
                )
            col = render_column_strength_bar(
                st.column_strength, w, margin_frac=args.edge_ignore
            )
            cv2.imshow("evaluate: Columns", col)
            cv2.imshow(
                "evaluate: Gate",
                cv2.resize(
                    disp,
                    (min(1400, disp.shape[1]), min(900, disp.shape[0])),
                ),
            )
            if cv2.waitKey(args.delay) == 27:
                break

        if st.state == "two":
            n_two += 1
            if pose is not None and pose.ok:
                reprojs.append(pose.reproj_err_px)
                if pose.reproj_err_px <= 25.0:
                    yaw_ok.append(abs(pose.yaw_error_deg))
        elif st.state == "one":
            n_one += 1
        else:
            n_none += 1

    if args.viz:
        cv2.destroyAllWindows()

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
    print("- For live: live_gate_detector.py --edge-ignore 0.05 --temporal 0.35")


if __name__ == "__main__":
    main()
