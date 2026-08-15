"""
Navigation gate: two poles + center line + optional PnP (normal / yaw hint).

  python3 scripts/navigation_gate_detector.py
  python3 scripts/navigation_gate_detector.py --temporal 0.35 --fov 65
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gate_detection.pipeline import PipelineConfig
from scripts.batch_viewer import add_common_args, resolve_folder, run_batch


def main() -> None:
    p = argparse.ArgumentParser(description="Navigation gate detector")
    add_common_args(p, "images/image_navigation_01")
    args = p.parse_args()

    cfg = PipelineConfig(
        use_color_boost=False,
        horizontal_fov_deg=args.fov,
        gate_width_m=args.gate_width,
        use_pnp=not args.no_pnp,
        temporal_alpha=args.temporal,
    )
    run_batch(
        folder=resolve_folder(args.folder),
        cfg=cfg,
        delay_ms=args.delay,
        no_gui=args.no_gui,
    )


if __name__ == "__main__":
    main()
