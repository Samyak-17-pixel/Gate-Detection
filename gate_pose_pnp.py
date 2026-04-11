"""
PnP gate pose: 4 image corners from pole lines + horizontal edges -> R,t -> plane normal in camera frame.

Gate model: rectangle in plane Z=0, X along width (m), Y up (m), origin at bottom-left corner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# SAUVC: pole spacing (m). Height often differs; we infer aspect from image quad by default.
GATE_WIDTH_M = 1.5
GATE_HEIGHT_M = 1.5


def camera_matrix_from_fov(
    width_px: int, height_px: int, horizontal_fov_deg: float = 60.0
) -> np.ndarray:
    """Approximate K from horizontal FOV (pinhole)."""
    fx = (width_px / 2.0) / np.tan(np.radians(horizontal_fov_deg) / 2.0)
    fy = fx
    cx, cy = width_px / 2.0, height_px / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=np.float64)


def object_points_rectangle(width_m: float, height_m: float) -> np.ndarray:
    """4x3: bottom-left, bottom-right, top-right, top-left (meters)."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [width_m, 0.0, 0.0],
            [width_m, height_m, 0.0],
            [0.0, height_m, 0.0],
        ],
        dtype=np.float64,
    )


def infer_gate_height_m(
    corners_ordered: List[Tuple[float, float]], width_m: float
) -> float:
    """Match 3D rectangle aspect to image quad (y-down): height_m = width_m * (ph / pw)."""
    if len(corners_ordered) != 4:
        return GATE_HEIGHT_M
    bl, br, tr, tl = (np.array(c, dtype=np.float64) for c in corners_ordered)
    pw = float(np.linalg.norm(br - bl))
    ph = float(np.linalg.norm(tl - bl))
    if pw < 1e-3:
        return GATE_HEIGHT_M
    return float(width_m * ph / pw)


def order_image_corners_quad(
    corners: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """
    Reorder 4 points to [BL, BR, TR, TL] in image coords (y down, x right).
    """
    if len(corners) != 4:
        return corners
    pts = np.array(corners, dtype=np.float64)
    yi = np.argsort(pts[:, 1])
    top_two = pts[yi[:2]]
    bot_two = pts[yi[2:]]
    top_two = top_two[np.argsort(top_two[:, 0])]
    bot_two = bot_two[np.argsort(bot_two[:, 0])]
    tl, tr = top_two[0], top_two[1]
    bl, br = bot_two[0], bot_two[1]
    return [(float(bl[0]), float(bl[1])), (float(br[0]), float(br[1])), (float(tr[0]), float(tr[1])), (float(tl[0]), float(tl[1]))]


def _fit_line_from_mask_points(
    mask: np.ndarray, x0: int, x1: int, y0: int, y1: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """fitLine on points where mask>0 in band [x0,x1]x[y0,y1]. Returns (point 2x1, direction 2x1)."""
    roi = mask[y0:y1, x0:x1]
    ys, xs = np.where(roi > 0)
    if ys.size < 30:
        return None
    pts = np.stack([xs + x0, ys + y0], axis=1).astype(np.float32)
    pts = pts.reshape(-1, 1, 2)
    line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x, y = float(line[0]), float(line[1]), float(line[2]), float(line[3])
    return np.array([[x], [y]], dtype=np.float64), np.array([[vx], [vy]], dtype=np.float64)


def _intersect_lines(
    p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray
) -> Optional[Tuple[float, float]]:
    x1, y1 = float(p1[0, 0]), float(p1[1, 0])
    vx1, vy1 = float(d1[0, 0]), float(d1[1, 0])
    x2, y2 = float(p2[0, 0]), float(p2[1, 0])
    vx2, vy2 = float(d2[0, 0]), float(d2[1, 0])
    det = vx1 * vy2 - vy1 * vx2
    if abs(det) < 1e-8:
        return None
    dx, dy = x2 - x1, y2 - y1
    t = (dx * vy2 - dy * vx2) / det
    return x1 + t * vx1, y1 + t * vy1


def _horizontal_line_from_edges(
    gray_roi: np.ndarray, x_left: int, x_right: int, prefer_upper: bool
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Fit horizontal-ish line from Sobel-y edges between poles."""
    h, w = gray_roi.shape[:2]
    x0 = max(0, min(x_left, x_right) - 5)
    x1 = min(w, max(x_left, x_right) + 5)
    band = gray_roi[:, x0:x1]
    blur = cv2.GaussianBlur(band, (5, 5), 0)
    sy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    sy = np.uint8(np.absolute(sy))
    e = cv2.Canny(sy, 25, 70)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3))
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, k)
    row_s = np.sum(e, axis=1)
    if np.max(row_s) < 50:
        return None
    row_s = cv2.GaussianBlur(row_s.reshape(-1, 1), (31, 1), 0).flatten()
    peaks = np.argsort(row_s)[::-1][:12]
    if prefer_upper:
        half = h // 2
        peaks = sorted([p for p in peaks if p < half], key=lambda p: row_s[p], reverse=True)
        if not peaks:
            peaks = [int(np.argmax(row_s[:half]))]
    else:
        half = h // 2
        peaks = sorted([p for p in peaks if p >= half], key=lambda p: row_s[p], reverse=True)
        if not peaks:
            peaks = [half + int(np.argmax(row_s[half:]))]
    y_line = int(peaks[0])
    pts = np.column_stack(
        [np.arange(x0, x1, dtype=np.float32), np.full(x1 - x0, float(y_line), dtype=np.float32)]
    )
    pts = pts.reshape(-1, 1, 2)
    line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x, y = float(line[0]), float(line[1]), float(line[2]), float(line[3])
    return np.array([[x], [y]], dtype=np.float64), np.array([[vx], [vy]], dtype=np.float64)


@dataclass
class GatePoseResult:
    ok: bool
    rvec: Optional[np.ndarray]
    tvec: Optional[np.ndarray]
    """Unit plane normal in camera frame (gate plane; points from gate toward camera)."""
    normal_cam: Optional[np.ndarray]
    """Signed yaw hint (deg): positive ~ turn right to square up (camera +Z forward, level)."""
    yaw_error_deg: float
    reproj_err_px: float
    image_points: Optional[np.ndarray]
    corners_2d: Optional[List[Tuple[float, float]]]


def estimate_gate_pose(
    frame_bgr: np.ndarray,
    left_x: float,
    right_x: float,
    filtered_roi: np.ndarray,
    roi_y0: int,
    K: np.ndarray,
    dist: Optional[np.ndarray] = None,
    width_m: float = GATE_WIDTH_M,
    height_m: Optional[float] = None,
) -> GatePoseResult:
    """
    filtered_roi: binary vertical-edge mask in ROI coordinates (same as gate_detection_core).
    """
    H, W = frame_bgr.shape[:2]
    if dist is None:
        dist = np.zeros(5, dtype=np.float64)

    li, ri = int(round(left_x)), int(round(right_x))
    if ri <= li + 10:
        return GatePoseResult(
            False, None, None, None, 0.0, 0.0, None, None
        )

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_roi = gray[roi_y0 : roi_y0 + filtered_roi.shape[0], :]

    h_r, w_r = filtered_roi.shape
    bw = 10
    y_top, y_bot = 0, h_r

    left_line = _fit_line_from_mask_points(
        filtered_roi, max(0, li - bw), min(w_r, li + bw + 1), y_top, y_bot
    )
    right_line = _fit_line_from_mask_points(
        filtered_roi, max(0, ri - bw), min(w_r, ri + bw + 1), y_top, y_bot
    )

    top_h = _horizontal_line_from_edges(gray_roi, li, ri, prefer_upper=True)
    bot_h = _horizontal_line_from_edges(gray_roi, li, ri, prefer_upper=False)

    corners_img: List[Tuple[float, float]] = []

    if left_line and right_line and top_h and bot_h:
        lp, ld = left_line
        rp, rd = right_line
        tp, td = top_h
        bp, bd = bot_h
        bl = _intersect_lines(lp, ld, bp, bd)
        br = _intersect_lines(rp, rd, bp, bd)
        tr = _intersect_lines(rp, rd, tp, td)
        tl = _intersect_lines(lp, ld, tp, td)
        if all(c is not None for c in (bl, br, tr, tl)):
            for c in (bl, br, tr, tl):
                corners_img.append((float(c[0]), float(c[1]) + roi_y0))

    if len(corners_img) != 4:
        ys = []
        for xc in (li, ri):
            x0, x1 = max(0, xc - bw), min(w_r, xc + bw + 1)
            col = np.any(filtered_roi[:, x0:x1] > 0, axis=1)
            idx = np.where(col)[0]
            if idx.size > 0:
                ys.extend([int(idx.min()), int(idx.max())])
        if not ys:
            return GatePoseResult(False, None, None, None, 0.0, 0.0, None, None)
        yt = float(min(ys) + roi_y0)
        yb = float(max(ys) + roi_y0)
        lx_f, rx_f = float(li), float(ri)
        corners_img = [
            (lx_f, yb),
            (rx_f, yb),
            (rx_f, yt),
            (lx_f, yt),
        ]

    corners_img = order_image_corners_quad(corners_img)
    hm = height_m
    if hm is None:
        hm = infer_gate_height_m(corners_img, width_m)
    obj_pts = object_points_rectangle(width_m, hm).astype(np.float32)
    img_pts_2d = np.array(corners_img, dtype=np.float32)
    Kd = K.astype(np.float64)

    ok, rvec, tvec, _inliers = cv2.solvePnPRansac(
        obj_pts,
        img_pts_2d,
        Kd,
        dist,
        iterationsCount=200,
        reprojectionError=8.0,
        confidence=0.995,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or rvec is None or tvec is None:
        ok2, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts_2d,
            Kd,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok2:
            return GatePoseResult(
                False, None, None, None, 0.0, 0.0, None, corners_img
            )

    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            obj_pts, img_pts_2d, Kd, dist, rvec, tvec
        )
    except cv2.error:
        pass

    R, _ = cv2.Rodrigues(rvec)
    n_obj = np.array([[0.0], [0.0], [1.0]], dtype=np.float64)
    n_cam = (R @ n_obj).reshape(3)
    n_cam = n_cam / (np.linalg.norm(n_cam) + 1e-9)

    proj, _ = cv2.projectPoints(
        obj_pts.reshape(-1, 1, 3), rvec, tvec, Kd, dist
    )
    proj = proj.reshape(-1, 2)
    ip = img_pts_2d.reshape(-1, 2)
    reproj_err = float(np.mean(np.linalg.norm(proj - ip, axis=1)))

    into = (-n_cam.reshape(3)).astype(np.float64)
    into /= np.linalg.norm(into) + 1e-9
    nx, ny, nz = float(into[0]), float(into[1]), float(into[2])
    yaw_error_deg = float(np.degrees(np.arctan2(nx, nz)))

    return GatePoseResult(
        ok=True,
        rvec=rvec,
        tvec=tvec,
        normal_cam=n_cam.reshape(3),
        yaw_error_deg=yaw_error_deg,
        reproj_err_px=reproj_err,
        image_points=img_pts_2d.reshape(-1, 1, 2),
        corners_2d=corners_img,
    )
