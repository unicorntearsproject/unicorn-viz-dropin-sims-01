"""sims-01 drop-in package for unicorn-viz.

Provides the SimShowcase effect which renders OpenUSD scenes (.usda / .usdc /
.usdz) as audio-reactive fullscreen 3D visuals.  When no scene files are
present a built-in spiral-galaxy particle fallback is used so the effect is
never blank.
"""
from __future__ import annotations

from .sim_showcase import SimShowcase

__all__ = ['SimShowcase']
