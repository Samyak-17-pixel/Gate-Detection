"""
Qualification gate: HSV red boost + poles + center + PnP.

  python3 scripts/qualification_gate_detector.py --folder images/image_qualification_01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gate_detection.pipeline import PipelineConfig
from scripts.batch_viewer import add_common_args, resolve_folder, run_batch


def main() -> None:
    p = argparse.ArgumentParser(description="Qualification gate detector")
    add_common_args(p, "images/image_qualification_01")
    args = p.parse_args()

    cfg = PipelineConfig(
        use_color_boost=True,
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
        extra_summary=True,
    )


if __name__ == "__main__":
    main()
