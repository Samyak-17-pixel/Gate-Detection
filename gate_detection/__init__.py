"""SAUVC gate detection: vertical poles, optional PnP pose, overlays."""

from .detection_core import (
    GateDetectionResult,
    GateState,
    GateStateResult,
    detect_gate,
    detect_gate_with_state,
)
from .pipeline import PipelineConfig, process_frame
from .pose_pnp import GatePoseResult, camera_matrix_from_fov, estimate_gate_pose
from .temporal import GateTemporalFilter

__all__ = [
    "GateDetectionResult",
    "GatePoseResult",
    "GateState",
    "GateStateResult",
    "GateTemporalFilter",
    "PipelineConfig",
    "camera_matrix_from_fov",
    "detect_gate",
    "detect_gate_with_state",
    "estimate_gate_pose",
    "process_frame",
]
