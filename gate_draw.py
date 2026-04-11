"""Overlays for gate detection (poles, center, PnP, status)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from gate_pose_pnp import GatePoseResult


def draw_status_corner(frame: np.ndarray, text: str, color: Tuple[int, int, int]) -> None:
    cv2.putText(
        frame,
        text,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_two_poles_and_center(
    frame: np.ndarray,
    left_x: float,
    right_x: float,
    center_x: float,
    center_y: float,
    distance_px: float,
    center_error_px: float,
    skew_score: float,
    pose: Optional[GatePoseResult] = None,
    bars_y: Optional[List[int]] = None,
) -> None:
    h, w = frame.shape[:2]
    li, ri = int(round(left_x)), int(round(right_x))
    cxi = int(round(center_x))

    cv2.line(frame, (li, 0), (li, h), (0, 200, 0), 5)
    cv2.line(frame, (ri, 0), (ri, h), (0, 200, 0), 5)
    cv2.line(frame, (cxi, 0), (cxi, h), (0, 255, 255), 3)

    ref_x = w // 2
    cv2.line(frame, (ref_x, h // 2 - 20), (ref_x, h // 2 + 20), (180, 180, 180), 1)

    cyi = int(np.clip(round(center_y), 0, h - 1))
    cv2.circle(frame, (cxi, cyi), 8, (0, 255, 255), 2)

    if bars_y:
        for y in bars_y:
            cv2.line(frame, (li, y), (ri, y), (255, 100, 0), 3)

    y0 = min(100, h // 10)
    msg = f"W={distance_px:.0f}px  err_x={center_error_px:+.0f}  skew={skew_score:+.2f}"
    cv2.putText(frame, msg, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    if pose and pose.ok and pose.normal_cam is not None:
        n = pose.normal_cam
        msg2 = (
            f"yaw_err={pose.yaw_error_deg:+.1f}deg  reproj={pose.reproj_err_px:.1f}px  "
            f"n=({n[0]:.2f},{n[1]:.2f},{n[2]:.2f})"
        )
        cv2.putText(
            frame, msg2, (10, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1, cv2.LINE_AA
        )
        tip_x = int(np.clip(cxi + 120 * n[0], 0, w - 1))
        tip_y = int(np.clip(cyi - 120 * n[1], 0, h - 1))
        cv2.arrowedLine(frame, (cxi, cyi), (tip_x, tip_y), (255, 180, 100), 2, tipLength=0.25)

    if pose and pose.corners_2d and len(pose.corners_2d) == 4:
        for pt in pose.corners_2d:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 6, (255, 0, 255), 2)


def draw_one_pole_searching(frame: np.ndarray, single_x: float) -> None:
    h, w = frame.shape[:2]
    sx = int(round(np.clip(single_x, 0, w - 1)))
    cv2.line(frame, (sx, 0), (sx, h), (0, 165, 255), 3)
    draw_status_corner(frame, "2nd pole…", (200, 200, 200))


def render_column_strength_bar(
    column_strength: np.ndarray,
    width: int,
    *,
    height: int = 140,
    margin_frac: float = 0.04,
) -> np.ndarray:
    """
    BGR image: normalized column-strength curve + shaded ignored side bands (debug).
    """
    s = np.asarray(column_strength, dtype=np.float64).flatten()
    if s.size == 0:
        return np.zeros((height, max(320, width), 3), dtype=np.uint8)
    smax = float(np.max(s))
    if smax <= 0:
        smax = 1.0
    w = s.size
    bar_w = max(400, min(1200, w))
    img = np.zeros((height, bar_w, 3), dtype=np.uint8)
    m = max(2, int(w * margin_frac))
    cv2.rectangle(img, (0, 0), (int(m * bar_w / w), height), (40, 25, 25), -1)
    cv2.rectangle(
        img,
        (int((w - m) * bar_w / w), 0),
        (bar_w - 1, height),
        (40, 25, 25),
        -1,
    )
    pts = []
    for i in range(w):
        x = int(i * (bar_w - 1) / max(1, w - 1))
        t = float(s[i]) / smax
        y = int((height - 14) * (1.0 - t)) + 6
        pts.append((x, y))
    if len(pts) >= 2:
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(
        img,
        "column strength (ignored edges shaded)",
        (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return img
