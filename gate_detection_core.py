"""
Shared underwater gate detection: vertical poles -> distance (px) and gate center.

Uses structure (Sobel-X + morphology + column histogram). Optional HSV red/orange
boost for qualification-style gates. Tuned for murky pool imagery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np

GateState = Literal["none", "one", "two"]


@dataclass
class GateDetectionResult:
    ok: bool
    left_x: float
    right_x: float
    center_x: float
    center_y: float
    distance_px: float
    """Binary map of long vertical edge support (for visualization)."""
    filtered_edges: Optional[np.ndarray] = None


@dataclass
class GateStateResult:
    """none: no pole; one: single pole (no center line); two: valid gate pair."""

    state: GateState
    left_x: float
    right_x: float
    center_x: float
    center_y: float
    distance_px: float
    single_x: float
    single_strength: float
    skew_score: float
    """ROI row offset for filtered_edges (same as build_column_strength)."""
    roi_y0: int
    column_strength: np.ndarray = field(repr=False)
    filtered_edges: Optional[np.ndarray] = None


def _local_maxima_1d(s: np.ndarray, min_dist: int) -> List[int]:
    """Indices of strict local maxima with spacing >= min_dist (greedy by height)."""
    s = np.asarray(s, dtype=np.float64)
    n = len(s)
    if n < 3:
        return []
    candidates: List[int] = []
    for i in range(1, n - 1):
        if s[i] >= s[i - 1] and s[i] >= s[i + 1]:
            if i > 1 and s[i] == s[i - 1]:
                continue
            candidates.append(i)
    candidates.sort(key=lambda i: s[i], reverse=True)
    picked: List[int] = []
    for i in candidates:
        if all(abs(i - j) >= min_dist for j in picked):
            picked.append(i)
        if len(picked) >= 24:
            break
    return picked


def _parabolic_peak_x(s: np.ndarray, ix: int) -> float:
    """Sub-pixel x refinement; ix in [1, len-2]."""
    ix = int(np.clip(ix, 1, len(s) - 2))
    y0, y1, y2 = float(s[ix - 1]), float(s[ix]), float(s[ix + 1])
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-9:
        return float(ix)
    dx = 0.5 * (y0 - y2) / denom
    return float(ix + np.clip(dx, -0.5, 0.5))


def _vertical_run_filter(edges: np.ndarray, min_run: int) -> np.ndarray:
    """Keep vertical edge segments at least `min_run` tall (morphological opening)."""
    min_run = int(max(3, min_run))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_run))
    return cv2.morphologyEx(edges, cv2.MORPH_OPEN, k)


def _gate_center_y_from_runs(
    filtered: np.ndarray, left_i: int, right_i: int, half_w: int = 6
) -> float:
    """Vertical midpoint of strong responses near each pole column."""
    h = filtered.shape[0]
    xs = []
    for xc in (left_i, right_i):
        x0 = max(0, xc - half_w)
        x1 = min(filtered.shape[1], xc + half_w + 1)
        col = np.any(filtered[:, x0:x1] > 0, axis=1)
        ys = np.where(col)[0]
        if ys.size == 0:
            continue
        ys_min, ys_max = float(ys.min()), float(ys.max())
        xs.append(0.5 * (ys_min + ys_max))
    if not xs:
        return 0.5 * float(h)
    return float(np.mean(xs))


def _red_boost_mask_bgr(bgr: np.ndarray) -> np.ndarray:
    """Underwater red/orange: two hue ranges + moderate S/V floors."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 60, 50), (12, 255, 255))
    m2 = cv2.inRange(hsv, (165, 60, 50), (180, 255, 255))
    m = cv2.bitwise_or(m1, m2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return (m.astype(np.float32) / 255.0).clip(0.0, 1.0)


def build_column_strength(
    frame_bgr: np.ndarray,
    *,
    use_color_boost: bool = False,
    roi_top_frac: float = 0.02,
    roi_bot_frac: float = 0.98,
    edge_ignore_frac: float = 0.04,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Returns (column_strength 1d float, filtered binary uint8, (x0,y0,x1,y1) ROI in frame coords).

    ``edge_ignore_frac`` zeros the column histogram near left/right image borders so webcam
    letterboxing, bezels, and vignetting do not dominate as fake vertical poles.
    """
    height, width = frame_bgr.shape[:2]
    y0 = int(height * roi_top_frac)
    y1 = int(height * roi_bot_frac)
    roi = frame_bgr[y0:y1, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = np.uint8(np.absolute(sobelx))

    mag = sobelx.astype(np.float64)
    nz = mag[mag > 0]
    med = float(np.median(nz)) if nz.size > 0 else 25.0
    low = int(max(12, 0.5 * med))
    high = int(max(28, 1.2 * med))
    edges = cv2.Canny(sobelx, low, high)

    if use_color_boost:
        wmap = _red_boost_mask_bgr(roi)
        edges = np.clip(
            edges.astype(np.float32) * (0.2 + 0.8 * wmap), 0, 255
        ).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 21))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    rh = y1 - y0
    min_run = max(45, int(0.20 * rh))
    filtered = _vertical_run_filter(edges, min_run)

    col = np.sum(filtered, axis=0).astype(np.float64)
    col = cv2.GaussianBlur(col.reshape(1, -1), (1, min(71, max(5, width // 12) | 1)), 0).flatten()
    col = np.sqrt(col + 1e-6)
    margin = max(3, int(width * edge_ignore_frac))
    if margin * 2 < width:
        col[:margin] = 0.0
        col[-margin:] = 0.0
    return col, filtered, (0, y0, width, y1)


def pick_pole_pair(
    column_strength: np.ndarray,
    width: int,
    *,
    min_sep_px: int = 60,
    max_sep_px: Optional[int] = None,
    strength_floor_ratio: float = 0.14,
) -> Optional[Tuple[int, int, float, float]]:
    """
    Returns (left_idx, right_idx, left_x_sub, right_x_sub) or None.
    """
    s = np.asarray(column_strength, dtype=np.float64)
    if np.max(s) <= 0:
        return None
    smax = float(np.max(s))
    floor = strength_floor_ratio * smax
    if max_sep_px is None:
        max_sep_px = int(0.82 * width)

    min_sep = min(min_sep_px, max_sep_px - 5)
    if min_sep < 20:
        min_sep = 20

    peaks = _local_maxima_1d(s, min_dist=max(8, min_sep // 8))
    peaks = [p for p in peaks if s[p] >= floor]
    if len(peaks) < 2:
        floor = 0.08 * smax
        peaks = [p for p in _local_maxima_1d(s, min_dist=6) if s[p] >= floor]

    best = None
    best_score = -1.0
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            a, b = peaks[i], peaks[j]
            if a > b:
                a, b = b, a
            sep = b - a
            if sep < min_sep or sep > max_sep_px:
                continue
            score = float(s[a] * s[b]) * (1.0 + 2e-4 * sep)
            if score > best_score:
                best_score = score
                best = (a, b)

    if best is None:
        order = np.argsort(s)[::-1][: min(24, len(s))]
        for ii in range(len(order)):
            for jj in range(ii + 1, len(order)):
                a, b = int(order[ii]), int(order[jj])
                if a > b:
                    a, b = b, a
                sep = b - a
                if sep < min_sep or sep > max_sep_px:
                    continue
                score = float(s[a] * s[b])
                if score > best_score:
                    best_score = score
                    best = (a, b)

    if best is None:
        return None

    li, ri = best
    lx = _parabolic_peak_x(s, li)
    rx = _parabolic_peak_x(s, ri)
    return (li, ri, lx, rx)


def pole_skew_metric(
    filtered: np.ndarray, left_i: int, right_i: int, half_w: int = 7
) -> float:
    """[-1,1] approx: imbalance of vertical support left vs right pole."""
    h, w = filtered.shape
    l0, l1 = max(0, left_i - half_w), min(w, left_i + half_w + 1)
    r0, r1 = max(0, right_i - half_w), min(w, right_i + half_w + 1)
    sl = float(np.sum(filtered[:, l0:l1]))
    sr = float(np.sum(filtered[:, r0:r1]))
    t = sl + sr + 1e-6
    return (sl - sr) / t


def detect_gate_with_state(
    frame_bgr: np.ndarray,
    *,
    use_color_boost: bool = False,
    min_sep_frac: float = 0.11,
    max_sep_frac: float = 0.78,
    edge_ignore_frac: float = 0.04,
) -> GateStateResult:
    """
    Two poles only if separation in [min_sep_frac, max_sep_frac] * width.
    One pole: dominant peak with no second peak in that band above relative threshold.
    """
    h, w = frame_bgr.shape[:2]
    col, filtered, (_x0, y0, _x1, _y1) = build_column_strength(
        frame_bgr,
        use_color_boost=use_color_boost,
        edge_ignore_frac=edge_ignore_frac,
    )
    min_sep = max(20, int(min_sep_frac * w))
    max_sep = min(int(max_sep_frac * w), w - 5)

    smax = float(np.max(col)) if col.size else 0.0
    if smax <= 0:
        return GateStateResult(
            "none", 0.0, 0.0, w / 2.0, h / 2.0, 0.0, 0.0, 0.0, 0.0, y0, col, filtered
        )

    pair = pick_pole_pair(
        col, w, min_sep_px=min_sep, max_sep_px=max_sep, strength_floor_ratio=0.12
    )
    if pair is not None:
        li_idx, ri_idx, lx, rx = pair
        if lx > rx:
            lx, rx = rx, lx
            li_idx, ri_idx = ri_idx, li_idx
        cy_local = _gate_center_y_from_runs(filtered, li_idx, ri_idx)
        cx = 0.5 * (lx + rx)
        cy = y0 + cy_local
        sk = pole_skew_metric(filtered, li_idx, ri_idx)
        return GateStateResult(
            "two",
            float(lx),
            float(rx),
            float(cx),
            float(cy),
            float(rx - lx),
            0.0,
            0.0,
            float(sk),
            y0,
            col,
            filtered,
        )

    peaks = _local_maxima_1d(col, min_dist=max(10, min_sep // 6))
    peaks = [p for p in peaks if col[p] >= 0.10 * smax]
    peaks.sort(key=lambda p: col[p], reverse=True)
    if not peaks:
        return GateStateResult(
            "none", 0.0, 0.0, w / 2.0, h / 2.0, 0.0, 0.0, 0.0, 0.0, y0, col, filtered
        )

    best = peaks[0]
    partner_in_band = False
    for p in peaks[1:]:
        d = abs(p - best)
        if min_sep <= d <= max_sep and col[p] >= 0.18 * col[best]:
            partner_in_band = True
            break
    if partner_in_band:
        return GateStateResult(
            "none", 0.0, 0.0, w / 2.0, h / 2.0, 0.0, 0.0, 0.0, 0.0, y0, col, filtered
        )

    sx = _parabolic_peak_x(col, int(np.clip(best, 1, len(col) - 2)))
    cy_local = _gate_center_y_from_runs(filtered, int(round(sx)), int(round(sx)))
    cy = y0 + cy_local
    return GateStateResult(
        "one",
        0.0,
        0.0,
        float(sx),
        float(cy),
        0.0,
        float(sx),
        float(col[best]),
        0.0,
        y0,
        col,
        filtered,
    )


def detect_gate(
    frame_bgr: np.ndarray,
    *,
    use_color_boost: bool = False,
) -> GateDetectionResult:
    """
    Detect two vertical poles; gate center uses pole mid-heights in the filtered map.
    """
    st = detect_gate_with_state(frame_bgr, use_color_boost=use_color_boost)
    h, w = frame_bgr.shape[:2]
    if st.state != "two":
        return GateDetectionResult(
            ok=False,
            left_x=0.0,
            right_x=0.0,
            center_x=w / 2.0,
            center_y=h / 2.0,
            distance_px=0.0,
            filtered_edges=st.filtered_edges,
        )
    return GateDetectionResult(
        ok=True,
        left_x=st.left_x,
        right_x=st.right_x,
        center_x=st.center_x,
        center_y=st.center_y,
        distance_px=st.distance_px,
        filtered_edges=st.filtered_edges,
    )
