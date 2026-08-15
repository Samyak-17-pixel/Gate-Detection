"""Shared GUI/batch loop for navigation and qualification image folders."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2

from scripts._repo import REPO_ROOT, ensure_repo_on_path

ensure_repo_on_path()

from gate_detection.pipeline import PipelineConfig, process_frame
from gate_detection.temporal import GateTemporalFilter


def resolve_folder(folder: str) -> Path:
    path = Path(folder)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def run_batch(
    *,
    folder: Path,
    cfg: PipelineConfig,
    delay_ms: int,
    no_gui: bool,
    extra_summary: bool = False,
) -> None:
    if not folder.is_dir():
        print("Folder not found:", folder, file=sys.stderr)
        sys.exit(1)

    temporal = (
        GateTemporalFilter(cfg.temporal_alpha, cfg.temporal_max_jump_frac)
        if cfg.temporal_alpha > 0
        else None
    )

    images = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
    n_two = n_one = n_none = 0
    n_pnp = 0

    if not no_gui:
        cv2.namedWindow("Gate Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gate Detection", min(1600, 1200), min(900, 800))
        cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Edges", 640, 360)

    for img_name in images:
        path = str(folder / img_name)
        frame = cv2.imread(path)
        if frame is None:
            continue

        st, pose, disp = process_frame(frame, cfg, temporal)
        if st.state == "two":
            n_two += 1
            if pose and pose.ok:
                n_pnp += 1
        elif st.state == "one":
            n_one += 1
        else:
            n_none += 1

        if not no_gui:
            if st.filtered_edges is not None:
                cv2.imshow("Edges", cv2.resize(st.filtered_edges, (640, 360)))
            cv2.imshow(
                "Gate Detection",
                cv2.resize(disp, (min(1600, disp.shape[1]), min(900, disp.shape[0]))),
            )
            if cv2.waitKey(delay_ms) == 27:
                break

    if not no_gui:
        cv2.destroyAllWindows()

    n = n_two + n_one + n_none
    print("\nDetection Summary")
    print("---------------------------")
    print("Total Images:", n)
    print("Two poles:", n_two, "| One pole:", n_one, "| None:", n_none)
    if n_two > 0:
        print("PnP solves (raw ok):", n_pnp, "/", n_two)
    if extra_summary and n > 0:
        print("Two-pole rate:", round(100 * n_two / n, 2), "%")
    if temporal is not None:
        temporal.reset()


def add_common_args(parser: argparse.ArgumentParser, default_folder: str) -> None:
    parser.add_argument("--folder", default=default_folder)
    parser.add_argument("--delay", type=int, default=500)
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--fov", type=float, default=60.0, help="Horizontal FOV (deg), approximate K")
    parser.add_argument("--gate-width", type=float, default=1.5, help="Pole spacing (m)")
    parser.add_argument("--no-pnp", action="store_true")
    parser.add_argument(
        "--temporal",
        type=float,
        default=0.0,
        help="EMA alpha for pole smoothing (0=off). Use ~0.35 for video-like sequences.",
    )
