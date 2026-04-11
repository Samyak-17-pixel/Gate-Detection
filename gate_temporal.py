"""EMA smoothing for sequential frames (video / live)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SmoothedPoles:
    left_x: float
    right_x: float


class GateTemporalFilter:
    def __init__(self, alpha: float = 0.45):
        self.alpha = float(alpha)
        self._sl: Optional[float] = None
        self._sr: Optional[float] = None

    def reset(self) -> None:
        self._sl = self._sr = None

    def update_two(self, left_x: float, right_x: float) -> Tuple[float, float]:
        a = self.alpha
        if self._sl is None:
            self._sl, self._sr = left_x, right_x
        else:
            self._sl = a * self._sl + (1.0 - a) * left_x
            self._sr = a * self._sr + (1.0 - a) * right_x
        return self._sl, self._sr
