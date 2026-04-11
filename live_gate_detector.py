#!/usr/bin/env python3
"""
Live gate detection from a webcam / V4L2 device (navigation-style, no HSV boost).

  python3 live_gate_detector.py
  python3 live_gate_detector.py --device 2 --width 1280 --height 720 --fov 70
  python3 live_gate_detector.py --qualification   # red/orange boost
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from gate_pipeline import PipelineConfig, process_frame
from gate_temporal import GateTemporalFilter


def main() -> None:
    p = argparse.ArgumentParser(description="Live gate detector (camera stream)")
    p.add_argument("--device", type=int, default=0, help="cv2.VideoCapture index")
    p.add_argument("--width", type=int, default=0, help="Request capture width (0=default)")
    p.add_argument("--height", type=int, default=0, help="Request capture height")
    p.add_argument("--fov", type=float, default=60.0)
    p.add_argument("--gate-width", type=float, default=1.5)
    p.add_argument("--no-pnp", action="store_true")
    p.add_argument(
        "--temporal",
        type=float,
        default=0.4,
        help="EMA alpha for poles (recommended for live video)",
    )
    p.add_argument(
        "--qualification",
        action="store_true",
        help="Use HSV red/orange boost (qualification gate)",
    )
    args = p.parse_args()

    cfg = PipelineConfig(
        use_color_boost=args.qualification,
        horizontal_fov_deg=args.fov,
        gate_width_m=args.gate_width,
        use_pnp=not args.no_pnp,
        temporal_alpha=max(0.0, args.temporal),
    )
    temporal = (
        GateTemporalFilter(cfg.temporal_alpha) if cfg.temporal_alpha > 0 else None
    )

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print("Could not open camera:", args.device, file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    cv2.namedWindow("Gate Live", cv2.WINDOW_NORMAL)

    t0 = time.time()
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            st, pose, disp = process_frame(frame, cfg, temporal)
            dt = time.time() - t0
            fps = n / dt if dt > 0 else 0.0
            cv2.putText(
                disp,
                f"FPS ~{fps:.1f}  state={st.state}",
                (10, disp.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("Gate Live", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
